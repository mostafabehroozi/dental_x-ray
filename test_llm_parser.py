"""Offline tests for LLM-assisted parsing: the modes, the readers, the records and the guards.

Nothing here reaches a network. The parser model is a scripted client that answers from a rule and
records every request, so a test can assert both what came back and - just as important - that no
request was made at all.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import dental_eval as ev
import dental_pipeline as dp
import experiments as xp
import llm_api
import llm_parser as lp
import location_adapter as la
import report_writer as rw
from response_cache import ResponseCache

UPPER_LEFT = "the left posterior region of the upper dentition"
LOWER_LEFT = "the left posterior region of the lower dentition"


class ParserClient:
    """OpenAI-style client for the parser role: answers from a rule, records every request.

    `answer` is either a string (always that reply), a list (consumed in order) or a callable taking
    the user message. A reply may be a dict to set finish_reason/usage.
    """

    def __init__(self, answer):
        self.answer, self.requests = answer, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @property
    def calls(self) -> int:
        return len(self.requests)

    def user_text(self, index: int = -1) -> str:
        return self.requests[index]["messages"][-1]["content"]

    def _next(self, user: str):
        if callable(self.answer):
            return self.answer(user)
        if isinstance(self.answer, list):
            return self.answer[min(len(self.requests) - 1, len(self.answer) - 1)]
        return self.answer

    def _create(self, **request):
        self.requests.append(request)
        reply = self._next(request["messages"][-1]["content"])
        if isinstance(reply, str):
            reply = {"text": reply}
        message = SimpleNamespace(content=reply.get("text", ""), refusal=None)
        usage = SimpleNamespace(prompt_tokens=reply.get("prompt_tokens", 11),
                                completion_tokens=reply.get("completion_tokens", 7))
        return SimpleNamespace(choices=[SimpleNamespace(message=message,
                                                        finish_reason=reply.get("finish_reason", "stop"))],
                               usage=usage)


def service(answer="", *, global_mode=None, modes=None, parse_retries=1, cache=None):
    """A parser service whose model is a scripted client; returns (service, client)."""
    client = ParserClient(answer)
    model = lp.ParserModel(None, "key", "parser-model", parse_retries=parse_retries,
                           response_cache=cache, client=client, call_log="off")
    return lp.ParserService(lp.ParserPolicy(global_mode, modes), model), client


def decision_reply(answer: str) -> str:
    return json.dumps({"answer": answer, "evidence": "", "reason": "test"})


def location_reply(regions, unresolved: bool = False) -> str:
    return json.dumps({"regions": list(regions), "unresolved": unresolved, "evidence": "", "reason": "t"})


# ----------------------------------------------------------------------------
# Modes: the three of them, the global override, manual mode and the defaults
# ----------------------------------------------------------------------------
class ModeTests(unittest.TestCase):
    def test_documented_defaults(self):
        """Every stage has a default, and it is the one the design justifies in STAGES."""
        self.assertEqual(lp.DEFAULT_MODES, {
            "whole_image_decision": "code_then_llm",
            "region_decision": "code_then_llm",
            "rationale_location": "llm",
            "saved_answer_reconstruction": "code_then_llm",
            "spotlight_decision": "code_then_llm",
            "spotlight_location": "llm",
            "location_json": "code_then_llm",
            "report_json": "code_then_llm",
            "report_fidelity": "llm",
            "vote_fraction": "code_then_llm",
        })
        for stage, spec in lp.STAGES.items():
            self.assertIn(spec["default_mode"], lp.MODES, stage)
            for field in ("what", "why", "input"):
                self.assertTrue(spec[field].strip(), f"{stage} needs a documented {field}")

    def test_manual_mode_follows_each_stage(self):
        policy = lp.ParserPolicy(None, {"rationale_location": "code", "region_decision": "llm"})
        self.assertEqual(policy.mode("rationale_location"), ("code", "code"))
        self.assertEqual(policy.mode("region_decision"), ("llm", "llm"))
        # every stage not named keeps its own default
        self.assertEqual(policy.mode("whole_image_decision"), ("code_then_llm", "code_then_llm"))
        self.assertEqual(policy.mode("report_fidelity"), ("llm", "llm"))

    def test_global_mode_overrides_every_stage(self):
        for global_mode in lp.MODES:
            policy = lp.ParserPolicy(global_mode, {"rationale_location": "code", "report_json": "llm"})
            self.assertEqual(set(policy.resolved().values()), {global_mode}, global_mode)
            # the stage's own setting is remembered, and reported next to the mode that won
            self.assertEqual(policy.selected("rationale_location"), "code")
            self.assertEqual(policy.mode("rationale_location"), ("code", global_mode))
        self.assertFalse(lp.ParserPolicy("code").uses_llm())
        self.assertTrue(lp.ParserPolicy("llm").uses_llm())
        self.assertTrue(lp.ParserPolicy("code_then_llm").uses_llm())
        self.assertTrue(lp.ParserPolicy(None).uses_llm())  # manual defaults include "llm" stages

    def test_a_code_only_manual_policy_needs_no_model(self):
        policy = lp.ParserPolicy(None, {s: "code" for s in lp.STAGES})
        self.assertFalse(policy.uses_llm())
        self.assertEqual(lp.ParserService(policy, None).policy.llm_stages(), [])

    def test_invalid_modes_and_stages_are_refused(self):
        for bad in ("Code", "llm_then_code", "", 1):
            with self.assertRaises(ValueError):
                lp.ParserPolicy(bad)
        with self.assertRaises(ValueError):
            lp.ParserPolicy(None, {"whole_image_decision": "sometimes"})
        with self.assertRaises(ValueError):
            lp.ParserPolicy(None, {"no_such_stage": "code"})
        with self.assertRaises(ValueError):
            lp.ParserPolicy(None).mode("no_such_stage")
        with self.assertRaises(ValueError):  # a stage that needs a model, without one
            lp.ParserService(lp.ParserPolicy("llm"), None)

    def test_build_returns_a_code_only_service_without_a_spec(self):
        built = lp.build(None, "code")
        self.assertIsNone(built.model)
        self.assertFalse(built.policy.uses_llm())
        with self.assertRaises(ValueError):
            lp.build(None, "llm")
        with self.assertRaises(ValueError):
            lp.build({"provider": "openai"}, "llm")  # no model name


# ----------------------------------------------------------------------------
# Decisions: code, llm, code_then_llm, and everything that can go wrong
# ----------------------------------------------------------------------------
QUESTION = dp.questions_for("caries")[0]


class DecisionTests(unittest.TestCase):
    def test_code_success_never_calls_the_model(self):
        for mode in ("code", "code_then_llm"):
            parser, client = service(decision_reply("no"), global_mode=mode)
            outcome = parser.decision("whole_image_decision", "Yes\nCaries in the upper left.", QUESTION)
            self.assertEqual((outcome.value, outcome.error), ("yes", None), mode)
            self.assertEqual(client.calls, 0, f"{mode}: a valid code result must not cost a call")
            self.assertTrue(outcome.record["code_ok"])
            self.assertFalse(outcome.record["llm_attempted"])
            self.assertIsNone(outcome.record["fallback_reason"])

    def test_code_only_never_calls_the_model_even_when_it_fails(self):
        parser, client = service(decision_reply("yes"), global_mode="code")
        outcome = parser.decision("whole_image_decision", "Yes and no.", QUESTION)
        self.assertEqual((outcome.value, outcome.error), (None, "missing_or_ambiguous_decision"))
        self.assertEqual(client.calls, 0)
        self.assertFalse(outcome.record["llm_attempted"])

    def test_code_failure_then_a_successful_model_read(self):
        parser, client = service(decision_reply("yes"), global_mode="code_then_llm")
        outcome = parser.decision("whole_image_decision", "Caries is evident distally.", QUESTION,
                                  context="image=img1")
        self.assertEqual((outcome.value, outcome.error), ("yes", None))
        self.assertEqual(client.calls, 1)
        record = outcome.record
        self.assertEqual(record["code_error"], "missing_or_ambiguous_decision")
        self.assertEqual(record["fallback_reason"], "missing_or_ambiguous_decision")
        self.assertTrue(record["llm_used"])
        self.assertEqual(record["model"], "parser-model")
        self.assertEqual(record["original_text"], "Caries is evident distally.")
        self.assertIn("Caries is evident distally.", record["parser_input"]["user"])
        self.assertEqual(record["parser_response"], decision_reply("yes"))
        self.assertEqual((record["prompt_tokens"], record["completion_tokens"]), (11, 7))
        self.assertIsNotNone(record["latency_seconds"])
        self.assertFalse(record["cache_hit"])
        self.assertEqual(record["retries"], 0)
        self.assertEqual(record["selected_mode"], "code_then_llm")
        self.assertEqual(record["resolved_mode"], "code_then_llm")

    def test_code_failure_then_a_failed_model_read_stays_unresolved(self):
        parser, client = service(decision_reply("unresolved"), global_mode="code_then_llm")
        outcome = parser.decision("whole_image_decision", "Hard to say.", QUESTION)
        self.assertIsNone(outcome.value, "an unreadable decision never becomes a No")
        self.assertEqual(outcome.error, "missing_or_ambiguous_decision")
        self.assertIn("code:missing_or_ambiguous_decision", outcome.record["failure_reason"])
        self.assertIn("parser:unresolved_decision", outcome.record["failure_reason"])
        self.assertEqual(client.calls, 1, "a parser that says 'unresolved' is not asked again")

    def test_llm_mode_reads_directly_and_its_answer_wins(self):
        parser, client = service(decision_reply("no"), global_mode="llm")
        outcome = parser.decision("whole_image_decision", "Yes\nCaries in the upper left.", QUESTION)
        self.assertEqual(outcome.value, "no", "in llm mode the strict reader is not consulted")
        self.assertFalse(outcome.record["code_attempted"])
        self.assertIsNone(outcome.record["code_ok"])
        self.assertEqual(client.calls, 1)

    def test_truncation_is_a_real_failure_and_is_told_to_the_parser(self):
        parser, client = service(decision_reply("yes"), global_mode="code_then_llm")
        outcome = parser.decision("region_decision", "Yes\nCaries in the lower", QUESTION, truncated=True)
        self.assertEqual(outcome.record["code_error"], "truncated_output")
        self.assertEqual(outcome.value, "yes")
        self.assertIn(lp.TRUNCATED_NOTE, client.user_text())
        parser, client = service(decision_reply("yes"), global_mode="llm")
        parser.decision("region_decision", "Yes", QUESTION, truncated=False)
        self.assertIn(lp.COMPLETE_NOTE, client.user_text())

    def test_malformed_parser_output_is_retried_visibly_then_accepted(self):
        parser, client = service(["not json at all", decision_reply("no")], global_mode="llm",
                                 parse_retries=1)
        outcome = parser.decision("whole_image_decision", "...", QUESTION)
        self.assertEqual(outcome.value, "no")
        self.assertEqual(client.calls, 2)
        self.assertEqual(outcome.record["retries"], 1)
        self.assertEqual([a["error"] for a in outcome.record["parser_attempts"]],
                         ["parser_reply_not_json", None])
        self.assertIn("could not be read", client.user_text(), "the retry carries a format reminder")

    def test_malformed_parser_output_everywhere_stays_unresolved(self):
        parser, client = service("still not json", global_mode="llm", parse_retries=2)
        outcome = parser.decision("whole_image_decision", "...", QUESTION)
        self.assertIsNone(outcome.value)
        self.assertEqual(outcome.error, "parser_reply_not_json")
        self.assertEqual(client.calls, 3)
        self.assertEqual(outcome.record["retries"], 2)

    def test_an_empty_or_cut_off_parser_reply_is_named_as_such(self):
        parser, _ = service("", global_mode="llm", parse_retries=0)
        self.assertEqual(parser.decision("whole_image_decision", "x", QUESTION).error, "empty_parser_response")
        parser, _ = service({"text": '{"answer": "y', "finish_reason": "length"}, global_mode="llm",
                            parse_retries=0)
        self.assertEqual(parser.decision("whole_image_decision", "x", QUESTION).error, "truncated_parser_output")

    def test_a_parser_answer_outside_the_schema_is_refused(self):
        for reply in ('{"answer": "maybe"}', '{"answer": true}', '{"reason": "x"}', '[]'):
            parser, _ = service(reply, global_mode="llm", parse_retries=0)
            outcome = parser.decision("whole_image_decision", "x", QUESTION)
            self.assertIsNone(outcome.value, reply)
            self.assertIsNotNone(outcome.error, reply)

    def test_the_parser_is_given_the_text_and_the_question_and_nothing_else(self):
        parser, client = service(decision_reply("yes"), global_mode="llm")
        parser.decision("whole_image_decision", "Some reply.", QUESTION)
        user = client.user_text()
        self.assertIn("Some reply.", user)
        self.assertIn(QUESTION, user)
        for leak in ("ground truth", "annotation", "label", "the correct answer"):
            self.assertNotIn(leak, user.lower())

    def test_stage_names_are_checked(self):
        parser, _ = service(decision_reply("yes"), global_mode="llm")
        with self.assertRaises(ValueError):
            parser.decision("rationale_location", "x", QUESTION)
        with self.assertRaises(ValueError):
            parser.location("whole_image_decision", "x")


# ----------------------------------------------------------------------------
# Locations: several regions, paraphrases, and the difference between none and unresolved
# ----------------------------------------------------------------------------
class LocationTests(unittest.TestCase):
    def test_several_regions_in_one_reply(self):
        parser, client = service(location_reply(["lower-left", "lower-right", "upper-anterior"]),
                                 global_mode="llm")
        outcome = parser.location("rationale_location", "Caries in both lower posterior areas and the front.")
        self.assertEqual(outcome.value, ["upper-anterior", "lower-right", "lower-left"],
                         "regions come back in the pipeline's own cell order")
        self.assertEqual(client.calls, 1)

    def test_a_paraphrase_the_strict_reader_misses(self):
        text = "Yes. A carious lesion is present in the lower left quadrant, on the first molar."
        self.assertEqual(dp.extract_regions(text), [], "the strict reader needs the exact descriptor")
        parser, _ = service(location_reply(["lower-left"]), global_mode="llm")
        self.assertEqual(parser.location("rationale_location", text).value, ["lower-left"])

    def test_naming_no_region_is_an_answer_and_unresolved_is_not(self):
        parser, _ = service(location_reply([]), global_mode="llm")
        outcome = parser.location("rationale_location", "Yes. Caries is present.")
        self.assertEqual((outcome.value, outcome.error), ([], None))
        parser, _ = service(location_reply([], unresolved=True), global_mode="llm")
        outcome = parser.location("rationale_location", "Yes. Caries somewhere at the back.")
        self.assertIsNone(outcome.value, "an unreadable location never becomes the empty set")
        self.assertEqual(outcome.error, "unresolved_location")

    def test_code_then_llm_only_pays_when_the_reader_found_nothing(self):
        parser, client = service(location_reply(["lower-left"]), global_mode="code_then_llm")
        found = parser.location("rationale_location", f"Yes\nCaries in {UPPER_LEFT}.")
        self.assertEqual((found.value, client.calls), (["upper-left"], 0))
        missing = parser.location("rationale_location", "Yes\nCaries at the back on the lower left.")
        self.assertEqual((missing.value, client.calls), (["lower-left"], 1))
        self.assertEqual(missing.record["fallback_reason"], "missing_location")
        # a No answer is not expected to place anything, so silence is a real answer there
        parser, client = service(location_reply(["lower-left"]), global_mode="code_then_llm")
        quiet = parser.location("rationale_location", "No\nNothing seen.", expects_location=False)
        self.assertEqual((quiet.value, quiet.error, client.calls), ([], None, 0))

    def test_a_region_outside_the_vocabulary_is_refused(self):
        parser, _ = service(location_reply(["lower-left", "Q3-posterior"]), global_mode="llm",
                            parse_retries=0)
        outcome = parser.location("rationale_location", "x")
        self.assertEqual(outcome.error, "parser_reply_unknown_region")
        self.assertIsNone(outcome.value)

    def test_the_prompt_carries_the_vocabulary_and_the_side_convention(self):
        parser, client = service(location_reply([]), global_mode="llm")
        parser.location("rationale_location", "text", question=QUESTION)
        user = client.user_text()
        for cell in dp.CELLS:
            self.assertIn(cell, user)
        for descriptor in dp.DESCRIPTORS:
            self.assertIn(descriptor, user)
        self.assertIn("never flip it", user)
        self.assertIn(QUESTION, user)


# ----------------------------------------------------------------------------
# The parser model itself: counters, cache, retries, provenance
# ----------------------------------------------------------------------------
class ParserModelTests(unittest.TestCase):
    def test_calls_are_counted_under_their_own_role(self):
        parser, client = service(decision_reply("yes"), global_mode="llm")
        for _ in range(3):
            parser.decision("whole_image_decision", "x", QUESTION)
        self.assertEqual((parser.model.calls, client.calls), (3, 3))
        self.assertEqual(parser.model.call_log.role, "parser")
        self.assertEqual(parser.model.call_log.completion_tokens, 21)
        self.assertEqual(parser.usage["whole_image_decision"],
                         {"parses": 3, "code_ok": 0, "code_failed": 0, "llm_calls": 3, "llm_ok": 3,
                          "fallbacks": 0, "cache_hits": 0, "retries": 0, "unresolved": 0})

    def test_usage_since_reports_only_one_stretch(self):
        parser, _ = service(decision_reply("yes"), global_mode="llm")
        parser.decision("whole_image_decision", "x", QUESTION)
        snapshot = parser.usage_snapshot()
        parser.decision("whole_image_decision", "y", QUESTION)
        self.assertEqual(parser.usage_since(snapshot)["whole_image_decision"]["parses"], 1)
        self.assertEqual(parser.usage["whole_image_decision"]["parses"], 2)

    def test_an_identical_request_is_reused_from_the_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ResponseCache(Path(tmp), {"role": "parser", "model": "parser-model"})
            parser, client = service(decision_reply("yes"), global_mode="llm", cache=cache)
            first = parser.decision("whole_image_decision", "same text", QUESTION)
            second = parser.decision("whole_image_decision", "same text", QUESTION)
            other = parser.decision("whole_image_decision", "other text", QUESTION)
            self.assertEqual(client.calls, 2, "the repeat is served from disk, the new text is not")
            self.assertFalse(first.record["cache_hit"])
            self.assertTrue(second.record["cache_hit"])
            self.assertFalse(other.record["cache_hit"])
            self.assertEqual(second.value, "yes")
            self.assertEqual(parser.model.cache_hits, 1)
            self.assertEqual(parser.usage["whole_image_decision"]["cache_hits"], 1)

    def test_transport_failures_use_the_visible_api_retries(self):
        class Flaky:
            def __init__(self):
                self.attempts = 0
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

            def _create(self, **_request):
                self.attempts += 1
                if self.attempts == 1:
                    raise RuntimeError("connection reset")
                message = SimpleNamespace(content=decision_reply("no"), refusal=None)
                return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                                       usage=None)

        client = Flaky()
        model = lp.ParserModel(None, "k", "m", api_call_retries=1, client=client, call_log="off")
        parser = lp.ParserService(lp.ParserPolicy("llm"), model)
        self.assertEqual(parser.decision("whole_image_decision", "x", QUESTION).value, "no")
        self.assertEqual(client.attempts, 2)

    def test_settings_carry_the_prompts_and_public_leaves_them_out(self):
        parser, _ = service(decision_reply("yes"), global_mode=None)
        settings = parser.settings()
        self.assertEqual(settings["policy"]["global_mode"], None)
        self.assertEqual(settings["policy"]["resolved_modes"], lp.DEFAULT_MODES)
        self.assertEqual(settings["policy"]["defaults"], lp.DEFAULT_MODES)
        self.assertIn("prompts", settings["model"])
        self.assertEqual(set(settings["stages"]), set(lp.STAGES))
        self.assertNotIn("prompts", parser.public()["model"])
        json.dumps(settings)  # every manifest must be able to hold it

    def test_the_fingerprint_follows_the_model_the_prompts_and_the_modes(self):
        first, _ = service(decision_reply("yes"), global_mode="llm")
        same, _ = service(decision_reply("yes"), global_mode="llm")
        self.assertEqual(first.fingerprint(), same.fingerprint())
        other_mode, _ = service(decision_reply("yes"), global_mode="code_then_llm")
        self.assertNotEqual(first.fingerprint(), other_mode.fingerprint())
        other_model = lp.ParserService(lp.ParserPolicy("llm"),
                                       lp.ParserModel(None, "k", "another-model", client=ParserClient("")))
        self.assertNotEqual(first.fingerprint(), other_model.fingerprint())

    def test_invalid_model_settings_are_refused(self):
        with self.assertRaises(ValueError):
            lp.ParserModel(None, "k", "m", token_param="max_new_tokens", client=ParserClient(""))
        with self.assertRaises(ValueError):
            lp.ParserModel(None, "k", "m", parse_retries=-1, client=ParserClient(""))
        with self.assertRaises(ValueError):
            lp.ParserModel(None, "k", "m", api_call_retries=-1, client=ParserClient(""))

    def test_from_api_reads_the_provider_registry(self):
        providers = {"acme": {"base_url": "https://acme.example/v1", "api_key": "acme-key"}}
        with mock.patch.dict(llm_api.PROVIDERS, {}, clear=True):
            llm_api.configure_providers(providers)
            model = lp.ParserModel.from_api({"provider": "acme", "model": "reader-1",
                                             "token_param": "max_completion_tokens", "temperature": None,
                                             "max_output_tokens": 999, "parse_retries": 2,
                                             "api_call_retries": 3, "request_options": {"seed": 1}},
                                            client=ParserClient(""))
        self.assertEqual((model.model, model.base_url), ("reader-1", "https://acme.example/v1"))
        self.assertEqual((model.token_param, model.temperature, model.max_output_tokens),
                         ("max_completion_tokens", None, 999))
        self.assertEqual((model.parse_retries, model.api_call_retries), (2, 3))
        self.assertEqual(model.request_options, {"seed": 1})
        self.assertNotIn("api_key", json.dumps(model.public()))


# ----------------------------------------------------------------------------
# The location adapter's JSON, its schema, and its ground-truth boundary
# ----------------------------------------------------------------------------
GOOD_UNITS = '{"boxes": [{"id": 1, "units": ["Q1-posterior"], "teeth": [16]}]}'


class LocationJsonTests(unittest.TestCase):
    def reply(self, text, truncated=False):
        return {"text": text, "finish_reason": "length" if truncated else "stop", "truncated": truncated}

    def test_valid_json_is_never_sent_to_the_parser(self):
        parser, client = service("", global_mode="code_then_llm")
        outcome = parser.location_json(GOOD_UNITS, 1, code=lambda: la.parse_units_checked(GOOD_UNITS, 1))
        self.assertEqual(outcome.value[1]["units"], ["Q1-posterior"])
        self.assertEqual(client.calls, 0)

    def test_invalid_json_is_repaired_and_recorded(self):
        broken = "Here you go:\n```json\n{\"boxes\": [{\"id\": 1, \"units\": [\"Q1-posterior\"],}]}\n```"
        parser, client = service(json.dumps({"boxes": [{"id": 1, "units": ["Q1-posterior"], "teeth": []}]}),
                                 global_mode="code_then_llm")
        outcome = parser.location_json(broken, 1, code=lambda: la.parse_units_checked(broken, 1))
        self.assertEqual(outcome.value, {1: {"units": ["Q1-posterior"], "teeth": []}})
        self.assertEqual(client.calls, 1)
        self.assertEqual(outcome.record["code_error"], "invalid_json")
        self.assertIn("invalid_json", client.user_text())

    def test_a_missing_box_is_a_schema_failure_the_parser_may_fix(self):
        half = '{"boxes": [{"id": 1, "units": ["Q1-posterior"], "teeth": []}]}'
        parser, client = service(json.dumps({"boxes": [{"id": 1, "units": ["Q1-posterior"], "teeth": []},
                                                       {"id": 2, "units": ["Q3-anterior"], "teeth": [31]}]}),
                                 global_mode="code_then_llm")
        outcome = parser.location_json(half, 2, code=lambda: la.parse_units_checked(half, 2))
        self.assertEqual(outcome.record["code_error"], "box_id_mismatch")
        self.assertEqual(sorted(outcome.value), [1, 2])
        self.assertEqual(outcome.value[2]["teeth"], [31])

    def test_double_failure_keeps_what_the_strict_reader_recovered(self):
        half = '{"boxes": [{"id": 1, "units": ["Q1-posterior"], "teeth": []}]}'
        parser, _ = service("no json here", global_mode="code_then_llm", parse_retries=0)
        outcome = parser.location_json(half, 2, code=lambda: la.parse_units_checked(half, 2))
        self.assertEqual(outcome.error, "box_id_mismatch", "the run still sees the original failure")
        self.assertEqual(sorted(outcome.value), [1], "the partially usable result is not thrown away")

    def test_the_parser_may_not_invent_a_unit_or_a_box(self):
        for reply in ('{"boxes": [{"id": 1, "units": ["upper right"], "teeth": []}]}',
                      '{"boxes": [{"id": 9, "units": ["Q1-posterior"], "teeth": []}]}',
                      '{"boxes": [{"id": 1, "units": ["Q1-posterior"], "teeth": ["x"]}]}',
                      '{"boxes": [{"id": 1, "units": ["Q1-posterior"], "teeth": []},'
                      ' {"id": 1, "units": ["Q2-anterior"], "teeth": []}]}'):
            parser, _ = service(reply, global_mode="llm", parse_retries=0)
            outcome = parser.location_json("x", 1, code=lambda: ({}, "invalid_json"))
            self.assertIsNotNone(outcome.error, reply)

    def test_the_parser_never_sees_the_ground_truth_of_the_boxes(self):
        if importlib.util.find_spec("PIL") is None:
            self.skipTest("Pillow not installed")
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp, "img1.png")
            _blank(image)
            parser, client = service(json.dumps({"boxes": [{"id": 1, "units": ["Q1-posterior"], "teeth": []}]}),
                                     global_mode="code_then_llm")
            adapter = la.LLMAdapter(None, "k", "vision", parse_retries=0, parser=parser, call_log="off",
                                    client=ParserClient("not json"))
            boxes = [{"condition": "carious_lesion", "xc": 0.2, "yc": 0.3, "w": 0.1, "h": 0.1}]
            rows = adapter.adapt(image, boxes, "img1")
            self.assertEqual(rows[0]["regions"], dp.units_to_cells(["Q1-posterior"]))
            self.assertEqual(rows[0]["source"], "llm")
            user = client.user_text()
            self.assertNotIn("carious", user.lower())
            self.assertNotIn("Dental caries", user)
            self.assertIn("1 numbered bounding boxes", user.replace("classify 1 ", "1 "))
            for unit in dp.UNITS:
                self.assertIn(unit, user)
            self.assertTrue(rows[0]["parsing"], "the adapter keeps the parse record next to the box")


def _blank(path: Path, size=(560, 280)) -> None:
    from PIL import Image

    Image.new("L", size, color=128).save(path)


# ----------------------------------------------------------------------------
# The report: JSON extraction, vote fractions, semantic fidelity
# ----------------------------------------------------------------------------






# ----------------------------------------------------------------------------
# Fixtures shared by the report tests
# ----------------------------------------------------------------------------




# ----------------------------------------------------------------------------
# The pipeline end to end: records, unresolved states, provenance, resume, equivalence
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# The spotlight adapter
# ----------------------------------------------------------------------------
class SpotlightTests(unittest.TestCase):
    def setUp(self):
        if importlib.util.find_spec("PIL") is None:
            self.skipTest("Pillow not installed")
        self.tmp = tempfile.TemporaryDirectory()
        self.image = Path(self.tmp.name, "img1.png")
        _blank(self.image)
        self.boxes = [{"condition": "carious_lesion", "xc": 0.2, "yc": 0.3, "w": 0.1, "h": 0.1}]

    def tearDown(self):
        self.tmp.cleanup()

    class Runner:
        def __init__(self, text):
            self.text, self.calls = text, 0

        def settings(self):
            return {"model": "fake"}

        def ask(self, image, question):
            self.calls += 1
            return {"text": self.text, "finish_reason": "stop", "truncated": False}

    def test_a_paraphrased_spotlight_location_is_read_and_placed(self):
        runner = self.Runner("Yes. Caries on the lower left first molar.")
        parser, client = service(lambda user: location_reply(["lower-left"]) if "THE SIX REGIONS" in user
                                 else decision_reply("yes"), global_mode=None)
        adapter = la.FdmAdapter(runner, parser=parser)
        rows = adapter.adapt(self.image, self.boxes, "img1")
        self.assertEqual((rows[0]["regions"], rows[0]["source"]), (["lower-left"], "fdm"))
        self.assertEqual([r["stage"] for r in rows[0]["attempts"][0]["parsing"]],
                         ["spotlight_decision", "spotlight_location"])
        self.assertEqual(client.calls, 1, "the strict reader read the Yes; only the location cost a call")

    def test_a_no_answer_is_never_turned_into_a_region(self):
        runner = self.Runner(f"No. Nothing is seen in {LOWER_LEFT}.")
        parser, client = service(decision_reply("no"), global_mode="llm")
        rows = la.FdmAdapter(runner, parser=parser).adapt(self.image, self.boxes, "img1")
        self.assertIsNone(rows[0]["regions"], "the box falls back to geometry, as it always has")
        self.assertEqual(client.calls, 1, "a No answer is never put to the location parser")

    def test_an_unresolved_spotlight_location_falls_back_to_geometry(self):
        runner = self.Runner("Yes. Caries at the back.")
        parser, _ = service(lambda user: location_reply([], unresolved=True) if "THE SIX REGIONS" in user
                            else decision_reply("yes"), global_mode="llm", parse_retries=0)
        rows = la.FdmAdapter(runner, parser=parser).adapt(self.image, self.boxes, "img1")
        self.assertIsNone(rows[0]["regions"])
        self.assertEqual(rows[0]["fallback_reason"], "parse_exhausted_or_not_localized")

    def test_the_adapter_settings_carry_the_parser(self):
        parser, _ = service(decision_reply("yes"), global_mode="llm")
        settings = la.FdmAdapter(self.Runner("Yes"), parser=parser).settings()
        self.assertEqual(settings["parser"]["policy"]["global_mode"], "llm")
        code_only = la.FdmAdapter(self.Runner("Yes"), parser=lp.code_only()).settings()
        self.assertNotIn("parser", code_only, "code-only reading is the absence of a parser")


# ----------------------------------------------------------------------------
# Configuration: the knobs, their validation, and what reaches the summaries
# ----------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


# ----------------------------------------------------------------------------
# What an unresolved location becomes downstream: never a claim about a region
# ----------------------------------------------------------------------------


def unlocated_result() -> dict:
    """A result whose rationale genuinely named no region (the empty set, not unresolved)."""
    result = fake_result()
    result["tasks"]["caries"]["regions"] = []
    result["findings"] = {c: dp._finding(c, result["tasks"], dp.Protocol()) for c in dp.CONDITIONS}
    return result
