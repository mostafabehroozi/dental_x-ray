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
def minimal_report(**overrides) -> dict:
    report = {
        "title": "Report", "headings": {"image": "Image", "findings": "Findings", "impression": "Impression",
                                        "not_assessable": "Not assessable", "limitations": "Limitations"},
        "sections": [], "impression": ["Nothing of note."], "not_assessable": [],
        "limitations": ["Automated output."],
    }
    report.update(overrides)
    return report


class ReportParsingTests(unittest.TestCase):
    def setUp(self):
        self.result = fake_result()
        self.structured = rw.structured_findings(self.result, "DentVLM")
        self.report = full_report(self.structured)

    def test_valid_report_json_is_never_sent_to_the_parser(self):
        parser, client = service("", global_mode="code_then_llm")
        text = json.dumps(self.report)
        report, record = rw.read_report_json(text, parser)
        self.assertEqual(report["title"], self.report["title"])
        self.assertEqual(client.calls, 0)
        self.assertTrue(record["code_ok"])

    def test_broken_report_json_is_recovered_verbatim(self):
        broken = "Sure!\n{'title': 'Report',}"
        parser, client = service(json.dumps({"report": self.report, "unresolved": False, "reason": "t"}),
                                 global_mode="code_then_llm")
        report, record = rw.read_report_json(broken, parser)
        self.assertEqual(report["title"], self.report["title"])
        self.assertEqual(client.calls, 1)
        self.assertIn("never write a value the reply does not contain",
                      record["parser_input"]["user"].lower())

    def test_an_unrecoverable_report_stays_missing(self):
        parser, _ = service(json.dumps({"report": {}, "unresolved": True, "reason": "cut off"}),
                            global_mode="code_then_llm", parse_retries=0)
        report, record = rw.read_report_json("...", parser)
        self.assertIsNone(report, "a missing report is visible; an invented one is not")
        self.assertEqual(record["error"], "invalid_report_json")

    def test_report_json_stage_off_reads_with_code(self):
        parser, client = service("", global_mode="code")
        report, record = rw.read_report_json(json.dumps(self.report), parser)
        self.assertEqual(report["title"], self.report["title"])
        self.assertEqual(client.calls, 0)
        self.assertIsNone(record, "a code-only stage adds no parser record to the report")

    def test_semantic_fidelity_problems_reach_the_verification(self):
        parser, client = service(json.dumps({"faithful": False, "unresolved": False,
                                             "problems": ["carious_lesion: 'severe' is not in the data"],
                                             "reason": "t"}),
                                 global_mode="llm")
        problems, records = rw.verify_report_detailed(self.report, self.structured, parser)
        self.assertEqual(problems, ["carious_lesion: 'severe' is not in the data"])
        self.assertEqual(client.calls, 1)
        self.assertEqual(records[-1]["stage"], "report_fidelity")

    def test_an_unresolved_fidelity_check_neither_fails_nor_passes_the_report(self):
        parser, _ = service(json.dumps({"faithful": False, "problems": [], "unresolved": True,
                                        "reason": "cannot compare"}),
                            global_mode="llm", parse_retries=0)
        problems, records = rw.verify_report_detailed(self.report, self.structured, parser)
        self.assertEqual(problems, [], "an unresolved check never invents a problem")
        self.assertEqual(records[-1]["error"], "unresolved_fidelity")
        self.assertEqual(records[-1]["value"], None)

    def test_fidelity_never_hides_a_structural_problem(self):
        parser, _ = service(json.dumps({"faithful": True, "problems": [], "unresolved": False, "reason": "ok"}),
                            global_mode="llm")
        broken = json.loads(json.dumps(self.report))
        broken["sections"][0]["findings"][0]["status"] = "present"
        problems, _ = rw.verify_report_detailed(broken, self.structured, parser)
        self.assertTrue(any("status must stay" in p for p in problems))

    def test_fidelity_is_skipped_in_code_mode(self):
        parser, client = service("", global_mode="code")
        problems, records = rw.verify_report_detailed(self.report, self.structured, parser)
        self.assertEqual((problems, records, client.calls), ([], [], 0))

    def test_fidelity_sees_the_two_documents_and_no_ground_truth(self):
        parser, client = service(json.dumps({"faithful": True, "problems": [], "unresolved": False, "reason": "t"}),
                                 global_mode="llm")
        rw.verify_report_detailed(self.report, self.structured, parser)
        user = client.user_text()
        self.assertIn(self.report["title"], user)
        self.assertIn("carious_lesion", user)
        self.assertNotIn("annotated", user)
        self.assertNotIn("bounding box", user.lower())


class VoteFractionTests(unittest.TestCase):
    def test_the_strict_reader_reads_plain_fractions_and_flags_what_it_cannot(self):
        self.assertEqual(rw.quoted_votes_checked("2/3 of the wordings agreed", 3), ({(2, 3)}, None))
        self.assertEqual(rw.quoted_votes_checked("٢/٣ اتفاق داشتند", 3), ({(2, 3)}, None))
        self.assertEqual(rw.quoted_votes_checked("teeth 16, 17 and 18", 3), (set(), None))
        self.assertEqual(rw.quoted_votes_checked("no counts here", 3), (set(), None))
        self.assertEqual(rw.quoted_votes_checked("2 out of 3 wordings agreed", 3),
                         (set(), "unreadable_vote_claim"))
        self.assertEqual(rw.quoted_votes_checked("2⁄3 of the wordings", 3), (set(), "unreadable_vote_claim"))
        self.assertEqual(rw.quoted_votes_checked("2 out of 3 wordings agreed", 0), (set(), None))

    def test_an_unreadable_claim_is_handed_to_the_parser(self):
        parser, client = service(json.dumps({"votes": [[2, 3]], "unresolved": False, "reason": "t"}),
                                 global_mode="code_then_llm")
        outcome = parser.vote_fraction("2 out of 3 wordings agreed", 3,
                                       code=lambda: rw.quoted_votes_checked("2 out of 3 wordings agreed", 3))
        self.assertEqual(outcome.value, {(2, 3)})
        self.assertEqual(client.calls, 1)
        plain = parser.vote_fraction("2/3 agreed", 3, code=lambda: rw.quoted_votes_checked("2/3 agreed", 3))
        self.assertEqual((plain.value, client.calls), ({(2, 3)}, 1), "a readable fraction costs nothing")

    def test_an_unresolved_vote_reading_never_invents_an_accusation(self):
        result = fake_result(phrasings=2)
        structured = rw.structured_findings(result, "DentVLM", vote_agreement=True)
        report = full_report(structured)
        report["sections"][0]["findings"][0]["statement"] += " Two out of three wordings agreed."
        parser, _ = service(json.dumps({"votes": [], "unresolved": True, "reason": "cannot read"}),
                            global_mode="llm", parse_retries=0)
        problems, records = rw.verify_report_detailed(report, structured, parser)
        self.assertFalse([p for p in problems if "vote counts the data does not hold" in p])
        self.assertTrue(any(r["stage"] == "vote_fraction" for r in records))

    def test_a_count_the_data_does_not_hold_is_still_caught_through_the_parser(self):
        result = fake_result(phrasings=2)
        structured = rw.structured_findings(result, "DentVLM", vote_agreement=True)
        report = full_report(structured)
        report["impression"] = ["Caries reported by 5 of 2 wordings."]
        parser, _ = service(json.dumps({"votes": [[5, 2]], "unresolved": False, "reason": "t"}),
                            global_mode="llm")
        problems, _ = rw.verify_report_detailed(report, structured, parser)
        self.assertTrue(any("impression: quotes vote counts" in p for p in problems))


# ----------------------------------------------------------------------------
# Fixtures shared by the report tests
# ----------------------------------------------------------------------------
def fake_result(phrasings: int = 1) -> dict:
    """One saved analyzer result, as the pipeline writes it, without running anything."""
    answers = [{"answer": "yes", "regions": ["upper-left"]} for _ in range(phrasings)]
    tasks = {}
    for task in dp.Protocol(phrasings=phrasings).tasks():
        yes = task == "caries"
        tasks[task] = {"name": dp.task_name(task),
                       "answers": answers if yes else [{"answer": "no", "regions": []}] * phrasings,
                       "presence": "yes" if yes else "no",
                       "whole_image": "yes" if yes else "no",
                       "regions": ["upper-left"] if yes else None}
    findings = {c: dp._finding(c, tasks, dp.Protocol(phrasings=phrasings)) for c in dp.CONDITIONS}
    return {"image": "/tmp/img1.png", "image_id": "img1", "image_sha256": "abc",
            "protocol": {"phrasings": phrasings, "region_vote": "union", "location": "rationale",
                         "ask_untrained": False, "extra_tasks": True, "parse_retries": 0},
            "location_level": "rationale", "left_is_image_left": True, "tasks": tasks,
            "findings": findings, "calls": [], "call_count": len(tasks)}


def full_report(structured: dict) -> dict:
    """A report that passes every structural check of verify_report."""
    sections = []
    for key, (label, conditions) in rw.CATEGORIES.items():
        by_finding = {f["finding"]: f for f in structured["findings"]}
        sections.append({"category": key, "heading": label, "findings": [
            {"finding": c, "status": by_finding[c]["status"], "statement": f"The analysis reports {c}."}
            for c in conditions]})
    unparseable = structured["summary"]["unparseable"]
    return minimal_report(sections=sections,
                          not_assessable=[f"{c} could not be read." for c in unparseable])


# ----------------------------------------------------------------------------
# The pipeline end to end: records, unresolved states, provenance, resume, equivalence
# ----------------------------------------------------------------------------
class PipelineTests(unittest.TestCase):
    def setUp(self):
        if importlib.util.find_spec("PIL") is None:
            self.skipTest("Pillow not installed")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.image = self.root / "img1.png"
        _blank(self.image)
        self.images = {"img1": self.image}

    def tearDown(self):
        self.tmp.cleanup()

    def runner(self, script=None):
        from test_dental_pipeline import FakeRunner

        return FakeRunner(script or {})

    def test_code_mode_is_behaviour_equivalent_to_no_parser_at_all(self):
        script = {("presence", "fillings", None): f"Yes\nFillings in {UPPER_LEFT}.",
                  ("presence", "caries", None): "Yes and no.",
                  ("presence", "implant", None): "Yes\nAn implant is visible."}
        plain = dp.analyze_image(self.runner(script), self.image, dp.Protocol())
        parser, client = service(decision_reply("no"), global_mode="code")
        read = dp.analyze_image(self.runner(script), self.image, dp.Protocol(), parser=parser)
        self.assertEqual(client.calls, 0, "global mode 'code' must never call the parser model")
        self.assertEqual(plain["findings"], read["findings"])
        self.assertEqual(plain["tasks"], read["tasks"])
        self.assertEqual([c["text"] for c in plain["calls"]], [c["text"] for c in read["calls"]])
        # the records are additive: the decisions themselves are identical
        self.assertTrue(all("parsing" in call for call in read["calls"]))
        self.assertTrue(all(r["resolved_mode"] == "code" for call in read["calls"] for r in call["parsing"]))
        self.assertIsNone(read["calls"][0]["parsing"][0]["original_text"])

    def test_a_reply_the_strict_reader_cannot_read_is_recovered_and_recorded(self):
        script = {("presence", "caries", None): "Caries is evident in the lower left molars."}
        replies = {"answer": decision_reply("yes"), "regions": location_reply(["lower-left"])}
        parser, client = service(lambda user: replies["regions"] if "THE SIX REGIONS" in user
                                 else replies["answer"], global_mode=None)
        result = dp.analyze_image(self.runner(script), self.image, dp.Protocol(), parser=parser)
        caries = result["findings"]["carious_lesion"]
        self.assertEqual((caries["presence"], caries["regions"]), ("yes", ["lower-left"]))
        call = next(c for c in result["calls"] if c["task"] == "caries")
        stages = [r["stage"] for r in call["parsing"]]
        self.assertEqual(stages, ["whole_image_decision", "rationale_location"])
        self.assertEqual(call["parsing"][0]["fallback_reason"], "missing_or_ambiguous_decision")
        self.assertEqual(call["parsing"][1]["resolved_mode"], "llm")
        self.assertEqual(call["parsing"][1]["original_text"], script[("presence", "caries", None)])
        self.assertGreater(client.calls, 0)
        self.assertEqual(result["parser_usage"]["rationale_location"]["llm_ok"], 1)
        self.assertEqual(result["parser"]["policy"]["global_mode"], None)
        self.assertEqual(result["parser_fingerprint"], parser.fingerprint())

    def test_an_unresolved_location_stays_unresolved_and_is_left_out_of_scoring(self):
        script = {("presence", "caries", None): "Yes. Caries somewhere at the back."}
        parser, _ = service(lambda user: location_reply([], unresolved=True) if "THE SIX REGIONS" in user
                            else decision_reply("yes"), global_mode="llm", parse_retries=0)
        result = dp.analyze_image(self.runner(script), self.image, dp.Protocol(), parser=parser)
        caries = result["findings"]["carious_lesion"]
        self.assertEqual(caries["presence"], "yes")
        self.assertIsNone(caries["regions"], "an unreadable location never becomes an empty region set")
        self.assertIsNone(caries["region_count"])
        self.assertIsNone(ev.predicted_cells(result, "carious_lesion"),
                          "an unresolved location is excluded, not scored as six negatives")
        self.assertTrue(result["tasks"]["caries"]["answers"][0]["regions_unresolved"])

    def test_one_readable_task_still_places_a_finding_decided_by_two(self):
        tasks = {"prosthetic_crown": {"presence": "yes", "whole_image": "yes", "regions": ["upper-left"]},
                 "prosthetic_bridge": {"presence": "yes", "whole_image": "yes", "regions": None}}
        finding = dp._finding("prosthetic_restoration", tasks, dp.Protocol())
        self.assertEqual(finding["regions"], ["upper-left"])
        both_unresolved = {k: {"presence": "yes", "whole_image": "yes", "regions": None} for k in tasks}
        self.assertIsNone(dp._finding("prosthetic_restoration", both_unresolved, dp.Protocol())["regions"])

    def test_vote_leaves_out_an_unresolved_location(self):
        answers = [{"answer": "yes", "regions": [], "regions_unresolved": True},
                   {"answer": "yes", "regions": ["upper-left"]}]
        self.assertEqual(dp.vote(answers, "union"), {"presence": "yes", "regions": ["upper-left"]})
        all_unresolved = [{"answer": "yes", "regions": [], "regions_unresolved": True}]
        self.assertEqual(dp.vote(all_unresolved, "union"), {"presence": "yes", "regions": None})

    def test_the_manifest_holds_the_parser_and_a_changed_mode_stops_a_resume(self):
        parser, _ = service(decision_reply("yes"), global_mode="code_then_llm")
        out = self.root / "run"
        dp.run_dataset(self.runner(), self.images, out, dp.Protocol(), parser=parser)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["parser"]["policy"]["global_mode"], "code_then_llm")
        self.assertIn("prompts", manifest["parser"]["model"])
        other, _ = service(decision_reply("yes"), global_mode="llm")
        with self.assertRaises(ValueError):
            dp.run_dataset(self.runner(), self.images, out, dp.Protocol(), parser=other)
        with self.assertRaises(ValueError):  # and back to code-only is a different reading too
            dp.run_dataset(self.runner(), self.images, out, dp.Protocol(), parser=lp.code_only())

    def test_a_code_only_run_keeps_the_manifest_it_always_had(self):
        without = dp.run_config(dp.Protocol(), {"model": "fake"})
        with_code_only = dp.run_config(dp.Protocol(), {"model": "fake"}, parser=lp.code_only())
        self.assertEqual(without["hash"], with_code_only["hash"])
        self.assertNotIn("parser", with_code_only)

    def test_a_replay_under_another_parser_configuration_reads_with_code(self):
        saved = {"image_id": "img1", "parser_fingerprint": "0123456789abcdef", "calls": [
            {"stage": "region", "task": "caries", "cell": "lower-left", "question": "q?",
             "text": "Caries is evident here."}]}
        parser, client = service(decision_reply("yes"), global_mode="llm")
        self.assertEqual(dp.cell_answers(saved, parser), {"caries": {"lower-left": None}})
        self.assertEqual(client.calls, 0, "a foreign fingerprint is never reinterpreted")
        saved["parser_fingerprint"] = parser.fingerprint()
        self.assertEqual(dp.cell_answers(saved, parser), {"caries": {"lower-left": "yes"}})
        self.assertEqual(client.calls, 1)

    def test_an_accepted_answer_is_never_read_again(self):
        saved = {"image_id": "img1", "calls": [
            {"stage": "region", "task": "caries", "cell": "lower-left", "question": "q?",
             "text": "Caries is evident here.",
             "parse_recovery": {"attempt": 1, "value": None, "error": "missing_or_ambiguous_decision",
                                "status": "exhausted", "recovered": False, "max_attempts": 1}}]}
        parser, client = service(decision_reply("yes"), global_mode="llm")
        parser_fingerprint = parser.fingerprint()
        saved["parser_fingerprint"] = parser_fingerprint
        self.assertEqual(dp.cell_answers(saved, parser), {"caries": {"lower-left": None}})
        self.assertEqual(client.calls, 0, "the value the run accepted is the value, for good")


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
class ConfigurationTests(unittest.TestCase):
    def config(self, **overrides):
        return xp.resolve({"name": "x", **overrides},
                          {"backend": "local", "location_truth": "geometry",
                           "parser": {"api_key": "k"}})

    def test_the_shipped_default_reads_with_code_and_needs_no_parser_model(self):
        cfg = self.config()
        self.assertEqual(cfg["parser_mode"], "code")
        self.assertEqual(cfg["parser_modes"], lp.DEFAULT_MODES)
        self.assertFalse(xp.parser(cfg).policy.uses_llm())
        self.assertIsNone(xp.parser(cfg).model)

    def test_manual_mode_uses_the_per_stage_defaults(self):
        cfg = self.config(parser_mode=None)
        self.assertEqual(lp.ParserPolicy(cfg["parser_mode"], cfg["parser_modes"]).resolved(),
                         lp.DEFAULT_MODES)

    def test_one_stage_may_be_changed_without_losing_the_rest(self):
        cfg = self.config(parser_mode=None, parser_modes={"rationale_location": "code"})
        modes = lp.ParserPolicy(cfg["parser_mode"], cfg["parser_modes"]).resolved()
        self.assertEqual(modes["rationale_location"], "code")
        self.assertEqual(modes["report_fidelity"], "llm")

    def test_the_global_mode_wins_over_every_stage(self):
        cfg = self.config(parser_mode="llm", parser_modes={"rationale_location": "code"})
        modes = lp.ParserPolicy(cfg["parser_mode"], cfg["parser_modes"]).resolved()
        self.assertEqual(set(modes.values()), {"llm"})

    def test_bad_parser_settings_are_refused_before_the_sweep_starts(self):
        for bad in ({"parser_mode": "sometimes"}, {"parser_modes": {"nope": "code"}},
                    {"parser_modes": "code"}, {"parser_parse_retries": -1},
                    {"reuse_parser_responses": "yes"},
                    {"parser_mode": "llm", "parser": {"model": None}}):
            with self.assertRaises(ValueError, msg=bad):
                self.config(**bad)

    def test_the_retry_settings_reach_the_parser_spec(self):
        cfg = self.config(api_call_retries=5, parser_parse_retries=3)
        self.assertEqual((cfg["parser"]["api_call_retries"], cfg["parser"]["parse_retries"]), (5, 3))
        self.assertNotIn("api_key", json.dumps(xp.public(cfg)), "no key reaches experiment.json")

    def test_every_resolved_mode_appears_in_the_configuration_summary(self):
        lines = "\n".join(xp.parser_summary(self.config(parser_mode=None)))
        for stage, mode in lp.DEFAULT_MODES.items():
            self.assertIn(stage, lines)
        self.assertIn("manual mode", lines)
        self.assertIn("gpt-5", lines)
        self.assertIn("none (every stage reads with code)", "\n".join(xp.parser_summary(self.config())))

    def test_the_parser_cache_is_shared_and_namespaced(self):
        spec = {"base_url": "https://example/v1", "api_key": "k", "model": "reader-1"}
        with tempfile.TemporaryDirectory() as tmp:
            built = lp.build(spec, "llm", None, cache_root=tmp, client=ParserClient(""))
            cache = built.model.response_cache
            self.assertEqual(cache.root, Path(tmp) / "_parser_cache")
            self.assertIn("parser", cache.namespace["role"])
            # a different parser model is a different namespace, so nothing is reused across them
            other = lp.build({**spec, "model": "reader-2"}, "llm", None, cache_root=tmp,
                             client=ParserClient("")).model.response_cache
            self.assertNotEqual(cache.namespace_sha256, other.namespace_sha256)
        self.assertIsNone(lp.build(spec, "llm", None, client=ParserClient("")).model.response_cache)

    def test_the_notebook_configures_the_parser_role(self):
        notebook = json.loads(Path("main_notebook.ipynb").read_text(encoding="utf-8"))
        cells = ["".join(cell["source"]) for cell in notebook["cells"]]
        config = next(c for c in cells if "SHARED = {" in c)
        for knob in ("parser_mode", "parser_modes", "parser_parse_retries", "reuse_parser_responses"):
            self.assertIn(f'"{knob}"', config)
        self.assertIn('"parser": {"provider"', config)
        self.assertIn("import llm_parser as lp", next(c for c in cells if "import dental_pipeline as dp" in c))
        self.assertIn("llm_parser.py", next(c for c in cells if "REQUIRED_PROJECT_FILES" in c))
        self.assertIn("parser=parser", next(c for c in cells if "dp.run_dataset(" in c))
        self.assertIn("parser=open_parser(cfg)", next(c for c in cells if "xp.report_writer(" in c))
        self.assertIn("parser=open_parser(cfg)", next(c for c in cells if "xp.location_adapter(" in c))
        self.assertIn("parser_usage_comparison", next(c for c in cells if "CALL_USAGE_COMPARISON" in c))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


# ----------------------------------------------------------------------------
# What an unresolved location becomes downstream: never a claim about a region
# ----------------------------------------------------------------------------
class UnresolvedLocationTests(unittest.TestCase):
    def result(self):
        result = fake_result()
        result["tasks"]["caries"]["regions"] = None
        result["findings"] = {c: dp._finding(c, result["tasks"], dp.Protocol()) for c in dp.CONDITIONS}
        return result

    def test_the_structured_findings_say_unresolved_not_not_named(self):
        structured = rw.structured_findings(self.result(), "DentVLM")
        caries = next(f for f in structured["findings"] if f["finding"] == "carious_lesion")
        self.assertEqual(set(caries["regions"].values()), {"unresolved"})
        self.assertEqual(caries["located_in"], [])
        self.assertIn("could not be read", caries["location_status"])
        self.assertIn("unresolved", structured["legend"]["regions"])

    def test_a_readable_run_keeps_the_legend_it_always_had(self):
        structured = rw.structured_findings(fake_result(), "DentVLM")
        self.assertEqual(structured["legend"]["regions"], rw.LEGEND["regions (location from the rationale)"])

    def test_the_deterministic_summary_says_the_region_could_not_be_read(self):
        self.assertIn("region could not be read", dp.dentist_report(self.result()))
        self.assertIn("region not stated", dp.dentist_report(unlocated_result()))


def unlocated_result() -> dict:
    """A result whose rationale genuinely named no region (the empty set, not unresolved)."""
    result = fake_result()
    result["tasks"]["caries"]["regions"] = []
    result["findings"] = {c: dp._finding(c, result["tasks"], dp.Protocol()) for c in dp.CONDITIONS}
    return result


class CacheAndProvenanceTests(unittest.TestCase):
    def test_turning_the_cache_on_does_not_invalidate_a_resume(self):
        spec = {"base_url": "https://example/v1", "api_key": "k", "model": "reader-1"}
        with tempfile.TemporaryDirectory() as tmp:
            cached = lp.build(spec, "llm", None, cache_root=tmp, client=ParserClient(""))
            plain = lp.build(spec, "llm", None, client=ParserClient(""))
            self.assertEqual(cached.fingerprint(), plain.fingerprint(),
                             "reusing an identical reply is the same reading, not a different one")
            self.assertTrue(cached.public()["model"]["response_cache"])
            self.assertFalse(plain.public()["model"]["response_cache"])

    def test_a_changed_prompt_is_a_changed_configuration(self):
        before = service(decision_reply("yes"), global_mode="llm")[0].fingerprint()
        original = lp.PROMPTS["decision"]["system"]
        try:
            lp.PROMPTS["decision"]["system"] = original + " (edited)"
            edited = service(decision_reply("yes"), global_mode="llm")[0].fingerprint()
        finally:
            lp.PROMPTS["decision"]["system"] = original
        self.assertNotEqual(before, edited, "an edited prompt is a different reading of the same replies")

    def test_the_adapted_truth_of_two_readings_lives_in_two_directories(self):
        shared = {"backend": "local", "parser": {"api_key": "k"}}
        code = xp.resolve({"name": "a"}, shared)
        read = xp.resolve({"name": "b", "parser_mode": "code_then_llm"}, shared)
        self.assertNotEqual(xp.truth_dir(code, "d"), xp.truth_dir(read, "d"))
        same = xp.resolve({"name": "c"}, shared)
        self.assertEqual(xp.truth_dir(code, "d"), xp.truth_dir(same, "d"))

    def test_the_saved_configuration_holds_the_modes_that_actually_ran(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = xp.resolve({"name": "x", "parser_mode": "llm",
                              "parser_modes": {"rationale_location": "code"}},
                             {"backend": "local", "output_root": tmp,
                              "parser": {"api_key": "secret-parser-key"}})
            saved = json.loads(xp.record(cfg).read_text())
            self.assertEqual(saved["parser_modes"]["rationale_location"], "code")
            self.assertEqual(set(saved["parser_resolved_modes"].values()), {"llm"})
            self.assertNotIn("secret-parser-key", json.dumps(saved))
