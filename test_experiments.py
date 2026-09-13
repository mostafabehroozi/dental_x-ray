"""Offline tests for the experiment table: merging, validation, per-experiment paths and roles."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import experiments as xp

SHARED = {"output_root": "/tmp/out",
          "analyzer": {"provider": "openrouter", "model": "qwen/qwen3-vl", "api_key": "secret",
                       "request_options": {"extra_body": {"provider": {"only": ["novita"]}}}}}


class BuildTests(unittest.TestCase):
    def test_experiment_changes_only_what_it_lists(self):
        base, variant = xp.build([{"name": "base"}, {"name": "arch", "region_scheme": "arch"}], SHARED)
        self.assertEqual(base["region_scheme"], "quadrant")
        self.assertEqual(variant["region_scheme"], "arch")
        self.assertEqual(variant["analyzer"]["model"], "qwen/qwen3-vl")
        self.assertEqual(variant["presence_level"], base["presence_level"])

    def test_role_dictionary_merges_key_by_key(self):
        cfg, = xp.build([{"name": "swap", "analyzer": {"model": "gemini-3-pro"}}], SHARED)
        self.assertEqual(cfg["analyzer"]["model"], "gemini-3-pro")
        self.assertEqual(cfg["analyzer"]["provider"], "openrouter")
        self.assertEqual(cfg["analyzer"]["request_options"], SHARED["analyzer"]["request_options"])

    def test_unknown_knob_is_rejected_with_a_suggestion(self):
        with self.assertRaisesRegex(ValueError, "region_schema.*did you mean 'region_scheme'"):
            xp.build([{"name": "typo", "region_schema": "arch"}], SHARED)
        with self.assertRaisesRegex(ValueError, "shared: unknown knob"):
            xp.build([{"name": "base"}], {"outputroot": "/tmp/out"})

    def test_names_must_be_unique_and_directory_safe(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            xp.build([{"name": "same"}, {"name": "same"}], SHARED)
        for name in ("", "../escape", "with space"):
            with self.assertRaisesRegex(ValueError, "directory-safe"):
                xp.build([{"name": name}], SHARED)

    def test_question_form_auto_follows_the_backend(self):
        api, local = xp.build([{"name": "api"}, {"name": "local", "backend": "local"}], SHARED)
        self.assertEqual(api["question_form"], "combined")
        self.assertEqual(local["question_form"], "separate")
        self.assertEqual(xp.protocol(local).question_form, "separate")

    def test_invalid_values_fail_at_build_time(self):
        for bad, message in (({"backend": "gpu"}, "backend must be one of"),
                             ({"location_truth": "fdm"}, "needs backend 'local'"),
                             ({"presence_level": "image"}, "presence_level must be one of"),
                             ({"max_tokens": 0}, "max_tokens must be a positive integer"),
                             ({"report_images": 0}, "report_images must be None"),
                             ({"analyzer": {"model": None}}, "analyzer must be a dictionary")):
            with self.assertRaisesRegex(ValueError, message):
                xp.build([{"name": "bad", **bad}], SHARED)

    def test_retry_settings_reach_the_roles_and_specs_win(self):
        cfg, = xp.build([{"name": "base", "api_call_retries": 5, "location_failure_policy": "exclude",
                          "adapter": {"provider": "nvidia", "model": "m", "api_call_retries": 1}}], SHARED)
        self.assertEqual(cfg["analyzer"]["api_call_retries"], 5)
        self.assertEqual(cfg["adapter"]["api_call_retries"], 1)
        self.assertEqual(cfg["adapter"]["failure_policy"], "exclude")


class PathAndRoleTests(unittest.TestCase):
    def test_each_experiment_writes_to_its_own_directory(self):
        base, variant = xp.build([{"name": "base"}, {"name": "arch", "region_scheme": "arch"}], SHARED)
        self.assertEqual(xp.run_dir(base, "umfih_test"), Path("/tmp/out/base/umfih_test"))
        self.assertEqual(xp.run_dir(variant, "umfih_test"), Path("/tmp/out/arch/umfih_test"))

    def test_location_truth_is_shared_by_adapter_not_by_experiment(self):
        same, other, changed = xp.build([
            {"name": "a"}, {"name": "b", "region_scheme": "arch"},
            {"name": "c", "adapter": {"provider": "openai", "model": "gpt-5"}}], SHARED)
        self.assertEqual(xp.truth_dir(same, "d"), xp.truth_dir(other, "d"))
        self.assertNotEqual(xp.truth_dir(same, "d"), xp.truth_dir(changed, "d"))
        self.assertNotEqual(xp.truth_dir(same, "d"), xp.truth_dir(same, "other_dataset"))

    def test_provenance_and_public_never_carry_the_key(self):
        api, local = xp.build([{"name": "api"}, {"name": "local", "backend": "local"}], SHARED)
        self.assertNotIn("api_key", json.dumps(xp.provenance(api, llama_cpp_ref="b1")))
        self.assertEqual(xp.provenance(api, llama_cpp_ref="b1")["llama_cpp_ref"], "b1")
        self.assertNotIn("secret", json.dumps(xp.public(api), default=str))
        self.assertEqual(xp.provenance(local)["model_file"], xp.DEFAULTS["model_filename"])
        self.assertEqual(xp.analyzer_name(local), xp.DEFAULTS["model_filename"])
        self.assertEqual(xp.analyzer_name(api), "qwen/qwen3-vl")

    def test_record_writes_the_resolved_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, = xp.build([{"name": "base"}], {**SHARED, "output_root": tmp})
            saved = json.loads(xp.record(cfg).read_text(encoding="utf-8"))
            self.assertEqual(saved["name"], "base")
            self.assertNotIn("api_key", saved["analyzer"])

    def test_local_experiments_share_files_and_servers_by_their_settings(self):
        a, b, wide = xp.build([{"name": "a", "backend": "local"}, {"name": "b", "backend": "local",
                               "region_scheme": "arch"},
                               {"name": "wide", "backend": "local", "image_max_tokens": 4096}], SHARED)
        self.assertEqual(xp.server_key(a), xp.server_key(b))
        self.assertEqual(xp.model_key(a), xp.model_key(wide))
        self.assertNotEqual(xp.server_key(a), xp.server_key(wide))

    def test_local_runner_needs_a_server(self):
        local, = xp.build([{"name": "local", "backend": "local"}], SHARED)
        with self.assertRaisesRegex(ValueError, "needs a started llama.cpp server"):
            xp.runner(local)


class ModeTests(unittest.TestCase):
    def test_hosted_models_skip_the_probe(self):
        cfg, = xp.build([{"name": "api"}], SHARED)
        self.assertEqual(xp.resolve_mode(cfg), ("plain", None))

    def test_forced_mode_is_kept(self):
        cfg, = xp.build([{"name": "api", "mode": "tagged"}], SHARED)
        self.assertEqual(xp.resolve_mode(cfg), ("tagged", None))

    def test_a_resumed_run_reuses_the_mode_it_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, = xp.build([{"name": "local", "backend": "local"}], {**SHARED, "output_root": tmp})
            manifest = Path(tmp, "local", "umfih_test", "manifest.json")
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"mode": "tagged"}), encoding="utf-8")
            self.assertEqual(xp.resolve_mode(cfg, datasets=["umfih_test"]), ("tagged", None))

    def test_a_saved_probe_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, = xp.build([{"name": "local", "backend": "local"}], {**SHARED, "output_root": tmp})
            path = Path(tmp, "local", "probe.json")
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"recommended_mode": "plain"}), encoding="utf-8")
            mode, probe = xp.resolve_mode(cfg, datasets=["umfih_test"])
            self.assertEqual((mode, probe["recommended_mode"]), ("plain", "plain"))


if __name__ == "__main__":
    unittest.main()
