"""Mocked preflight checks; these do not establish real GPU/runtime equivalence."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import dental_pipeline as dp
import llama_runtime as lr


def server(tmp_path):
    value = lr.LlamaCppServer(tmp_path / "server", tmp_path / "model.gguf", tmp_path / "mmproj.gguf",
                              log_path=tmp_path / "server.log")
    value.provenance = {"launch_command": ["server", "--image-min-tokens", "4", "--image-max-tokens", "8192", "--n-predict", "512"]}
    return value


def props(value):
    return {"total_slots": 1, "model_path": str(value.model_path), "model_alias": value.alias,
            "modalities": {"vision": True}, "media_marker": "<__media_TEST__>", "chat_template": "fixture",
            "default_generation_settings": {"n_ctx": 16384, "params": {
                "temperature": .1, "top_p": .001, "repeat_penalty": 1.05, "repeat_last_n": -1,
                "seed": 0, "top_k": 0, "min_p": 0, "n_predict": -1,
                "samplers": ["penalties", "temperature", "top_p"]}}}


def response(data):
    return SimpleNamespace(json=lambda: data, raise_for_status=lambda: None)


def template(url, json, **kwargs):
    messages = json["messages"]
    assert len(messages) == 2 and messages[0]["content"] == dp.SYSTEM_MESSAGE
    content = messages[1]["content"]
    assert content[0]["type"] == "image_url"
    question = content[1]["text"]
    return response({"prompt": f"<|im_start|>system\n{dp.SYSTEM_MESSAGE}<|im_end|>\n"
                     f"<|im_start|>user\n<__media_TEST__>{question}<|im_end|>\n<|im_start|>assistant\n"})


def test_preflight_probes_real_multimodal_request_shape(tmp_path):
    value = server(tmp_path)
    with patch.object(lr.requests, "get", return_value=response(props(value))), \
         patch.object(lr.requests, "post", side_effect=template) as post:
        provenance = value.verify_runtime()
    assert post.call_count == 12
    assert len(provenance["rendered_templates"]) == 12
    assert all(prompt.count(dp.SYSTEM_MESSAGE) == 1 for prompt in provenance["rendered_templates"].values())
    assert "<__media_TEST__>" not in json.dumps(provenance)


@pytest.mark.parametrize("field,bad", [("top_p", .9), ("temperature", 0), ("repeat_last_n", 64),
    ("repeat_penalty", 1), ("seed", 1), ("top_k", 40), ("min_p", .05),
    ("samplers", ["penalties", "top_p", "temperature"])])
def test_mismatched_effective_sampling_blocks_inference(tmp_path, field, bad):
    value = server(tmp_path); payload = props(value)
    payload["default_generation_settings"]["params"][field] = bad
    with patch.object(lr.requests, "get", return_value=response(payload)), \
         pytest.raises(ValueError):
        value.verify_runtime()


@pytest.mark.parametrize("mutation", ["duplicate_system", "extra_instruction", "missing_image"])
def test_mismatched_template_blocks_inference(tmp_path, mutation):
    value = server(tmp_path)
    def wrong(*args, **kwargs):
        data = template(*args, **kwargs).json()
        if mutation == "duplicate_system": data["prompt"] = dp.SYSTEM_MESSAGE + data["prompt"]
        elif mutation == "extra_instruction": data["prompt"] += "Answer JSON"
        else: data["prompt"] = data["prompt"].replace("<__media_TEST__>", "")
        return response(data)
    with patch.object(lr.requests, "get", return_value=response(props(value))), \
         patch.object(lr.requests, "post", side_effect=wrong), pytest.raises(ValueError):
        value.verify_runtime()


def test_unidentified_healthy_server_cannot_be_reused(tmp_path):
    value = server(tmp_path)
    with patch.object(value, "_healthy", return_value=True), pytest.raises(ValueError, match="unidentified"):
        value.start()

def test_output_limit_is_checked_in_actual_process_arguments(tmp_path):
    value = server(tmp_path)
    value.provenance["launch_command"][-1] = "128"
    with patch.object(lr.requests, "get", return_value=response(props(value))), pytest.raises(ValueError, match="token limits"):
        value.verify_runtime()


def test_existing_artifact_requires_matching_provenance(tmp_path):
    files = lr.ModelFiles(tmp_path / "model.gguf", tmp_path / "mmproj.gguf")
    files.model_path.write_bytes(b"model"); files.mmproj_path.write_bytes(b"projector")
    with pytest.raises(ValueError, match="missing"):
        lr.verified_model_provenance(files)
    data = {"hf_repo": lr.DENTVLM_HF_REPO_ID, "hf_revision": lr.DENTVLM_REVISION, "outtype": "q8_0", "converter_revision": "pinned",
            "checkpoint_template_sha256": "fixture", "model_sha256": lr.sha256_file(files.model_path),
            "mmproj_sha256": lr.sha256_file(files.mmproj_path)}
    (tmp_path / lr.PROVENANCE_FILE).write_text(json.dumps(data))
    assert lr.verified_model_provenance(files) == data
    files.mmproj_path.write_bytes(b"another projector")
    with pytest.raises(ValueError, match="hash"):
        lr.verified_model_provenance(files)
