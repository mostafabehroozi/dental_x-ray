"""Offline tests for the experiment table: merging, validation, per-experiment paths and roles."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import experiments as xp

SHARED = {"output_root": "/tmp/out",
          "analyzer": {"provider": "openrouter", "model": "qwen/qwen3-vl", "api_key": "secret",
                       "request_options": {"extra_body": {"provider": {"only": ["novita"]}}}}}


class BuildTests(unittest.TestCase):
    def test_experiment_changes_only_what_it_lists(self):
        base, variant = xp.build([{"name": "base"}, {"name": "three", "phrasings": 3}], SHARED)
        self.assertEqual(base["phrasings"], 1)
        self.assertEqual(variant["phrasings"], 3)
        self.assertEqual(variant["analyzer"]["model"], "qwen/qwen3-vl")
        self.assertEqual(variant["location"], base["location"])

    def test_role_dictionary_merges_key_by_key(self):
        cfg, = xp.build([{"name": "swap", "analyzer": {"model": "gemini-3-pro"}}], SHARED)
        self.assertEqual(cfg["analyzer"]["model"], "gemini-3-pro")
        self.assertEqual(cfg["analyzer"]["provider"], "openrouter")
        self.assertEqual(cfg["analyzer"]["request_options"], SHARED["analyzer"]["request_options"])

    def test_unknown_knob_is_rejected_with_a_suggestion(self):
        with self.assertRaisesRegex(ValueError, "phrasing.*did you mean 'phrasings'"):
            xp.build([{"name": "typo", "phrasing": 3}], SHARED)
        with self.assertRaisesRegex(ValueError, "shared: unknown knob"):
            xp.build([{"name": "base"}], {"outputroot": "/tmp/out"})

    def test_response_reuse_switch_must_be_boolean(self):
        with self.assertRaisesRegex(ValueError, "reuse_local_responses must be True or False"):
            xp.build([{"name": "bad-cache", "reuse_local_responses": "yes"}], SHARED)

    def test_names_must_be_unique_and_directory_safe(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            xp.build([{"name": "same"}, {"name": "same"}], SHARED)
        for name in ("", "../escape", "with space"):
            with self.assertRaisesRegex(ValueError, "directory-safe"):
                xp.build([{"name": name}], SHARED)

    def test_invalid_values_fail_at_build_time(self):
        for bad, message in (({"backend": "gpu"}, "backend must be one of"),
                             ({"model_source": "torrent"}, "model_source must be one of"),
                             ({"location_truth": "fdm"}, "needs backend 'local'"),
                             ({"location": "boxes"}, "location must be one of"),
                             ({"phrasings": 9}, "phrasings must be between"),
                             ({"max_tokens": 0}, "max_tokens must be a positive integer"),
                             ({"smoke_images": -1}, "smoke_images must be a non-negative integer"),
                             ({"report_images": 0}, "report_images must be None"),
                             ({"analyzer": {"model": None}}, "analyzer must be a dictionary")):
            with self.assertRaisesRegex(ValueError, message):
                xp.build([{"name": "bad", **bad}], SHARED)

    def test_retry_settings_reach_the_roles_and_specs_win(self):
        cfg, = xp.build([{"name": "base", "api_call_retries": 5, "location_failure_policy": "exclude",
                          "adapter": {"provider": "openai", "model": "m", "api_call_retries": 1}}], SHARED)
        self.assertEqual(cfg["analyzer"]["api_call_retries"], 5)
        self.assertEqual(cfg["adapter"]["api_call_retries"], 1)
        self.assertEqual(cfg["adapter"]["failure_policy"], "exclude")

    def test_protocol_carries_every_question_knob(self):
        cfg, = xp.build([{"name": "loud", "phrasings": 3, "region_vote": "majority", "location": "regions",
                          "ask_untrained": True, "extra_tasks": False}], SHARED)
        protocol = xp.protocol(cfg)
        self.assertEqual((protocol.phrasings, protocol.region_vote, protocol.location), (3, "majority", "regions"))
        self.assertEqual((protocol.ask_untrained, protocol.extra_tasks), (True, False))


class PathAndRoleTests(unittest.TestCase):
    def test_each_experiment_writes_to_its_own_directory(self):
        base, variant = xp.build([{"name": "base"}, {"name": "three", "phrasings": 3}], SHARED)
        self.assertEqual(xp.run_dir(base, "umfih_test"), Path("/tmp/out/base/umfih_test"))
        self.assertEqual(xp.run_dir(variant, "umfih_test"), Path("/tmp/out/three/umfih_test"))

    def test_location_truth_is_shared_by_adapter_not_by_experiment(self):
        same, other, changed = xp.build([
            {"name": "a"}, {"name": "b", "phrasings": 3},
            {"name": "c", "adapter": {"provider": "gemini", "model": "gemini-3-pro"}}], SHARED)
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
        a, b, small = xp.build([{"name": "a", "backend": "local"},
                                {"name": "b", "backend": "local", "phrasings": 3},
                                {"name": "small", "backend": "local", "image_max_tokens": 1369}], SHARED)
        self.assertEqual(xp.server_key(a), xp.server_key(b))
        self.assertEqual(xp.model_key(a), xp.model_key(small))
        self.assertNotEqual(xp.server_key(a), xp.server_key(small))

    def test_local_runner_needs_a_server(self):
        local, = xp.build([{"name": "local", "backend": "local"}], SHARED)
        with self.assertRaisesRegex(ValueError, "needs a started llama.cpp server"):
            xp.runner(local)

    def test_local_response_cache_is_shared_only_by_matching_runtimes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = [root / name for name in ("model.gguf", "mmproj.gguf", "llama-server")]
            for path in files:
                path.write_bytes(path.name.encode())
            server = SimpleNamespace(model_path=files[0], mmproj_path=files[1], binary=files[2])
            base, phrased, smaller = xp.build([
                {"name": "base", "backend": "local"},
                {"name": "phrased", "backend": "local", "phrasings": 3},
                {"name": "smaller", "backend": "local", "image_max_tokens": 1369},
            ], {**SHARED, "output_root": tmp})

            base_cache = xp.local_response_cache(base, server)
            phrased_cache = xp.local_response_cache(phrased, server)
            smaller_cache = xp.local_response_cache(smaller, server)
            self.assertEqual(base_cache.namespace_sha256, phrased_cache.namespace_sha256)
            self.assertNotEqual(base_cache.namespace_sha256, smaller_cache.namespace_sha256)
            self.assertEqual(base_cache.root, Path(tmp, "_response_cache"))
            self.assertIsNone(xp.local_response_cache({**base, "reuse_local_responses": False}, server))

    def test_adapter_follows_the_location_knobs(self):
        off, windows = xp.build([{"name": "off", "evaluate_location": False},
                                 {"name": "windows", "location_truth": "geometry"}], SHARED)
        self.assertIsNone(xp.location_adapter(off))
        self.assertIsNone(xp.location_adapter(windows))


if __name__ == "__main__":
    unittest.main()
