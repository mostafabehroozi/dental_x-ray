"""Fixed configuration, migration and notebook contract."""
import ast
import json
from pathlib import Path
import pytest
import dental_pipeline as dp
import experiments as xp
import llm_parser as lp

@pytest.mark.parametrize("knob", sorted(xp.RETIRED))
def test_retired_switches_have_migration_messages(knob):
    with pytest.raises(ValueError, match="retired.*pan_training_aligned_v1"):
        xp.build([{"name": "old", knob: False}])

@pytest.mark.parametrize("override", [{"backend":"api"}, {"location":"regions"}, {"location_truth":"fdm"},
    {"max_tokens":513}, {"temperature":0}, {"ctx_size":8192}, {"image_min_tokens":None},
    {"image_max_tokens":1369}, {"parser_mode":"llm"}, {"parser_mode":None},
    {"reporter":{"include_rationale":True}}, {"reporter":{"vote_agreement":True}}])
def test_profile_rejects_distribution_changes(override):
    with pytest.raises(ValueError): xp.resolve({"name":"bad", **override})

def test_resolved_contract_and_manifest(tmp_path):
    cfg, = xp.build([{"name":dp.PROFILE}], {"output_root":str(tmp_path)})
    assert len(xp.protocol(cfg).tasks()) == 12
    assert not cfg["counting"] and cfg["smoke_images"] == 5
    assert not lp.ParserPolicy(cfg["parser_mode"], cfg["parser_modes"]).uses_llm()
    assert xp.location_adapter(cfg) is None
    saved = json.loads(xp.record(cfg).read_text())
    assert set(saved["parser_resolved_modes"].values()) == {"code"}
    assert cfg["hf_revision"] == "2ad8e71ea6708eee92723e7eca6e30e6dac48d85"
    assert xp.run_dir(cfg, "dataset") == tmp_path / dp.PROFILE / "dataset"

@pytest.mark.parametrize("config", [{"name":"../bad"}, {"name":"bad", "temperatur":.1},
                                    {"name":"bad", "model_source":"hf"}, {"name":"bad", "reuse_local_responses":"yes"}])
def test_bad_configuration_fails(config):
    with pytest.raises(ValueError): xp.resolve(config)

def test_the_notebook_resolves_fixed_parser_and_single_baseline():
    notebook = json.loads(Path("main_notebook.ipynb").read_text(encoding="utf8"))
    tree = ast.parse("".join(notebook["cells"][3]["source"]))
    env = {"xp":xp,"dp":dp,"OUTPUT_ROOT":"/tmp/test", "LOCATION_TRUTH":"geometry"}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(isinstance(t,ast.Name) and t.id in {"SHARED","EXPERIMENTS"} for t in statement.targets):
            exec(compile(ast.Module(body=[statement],type_ignores=[]),"cell3","exec"),env)
    assert len(env["EXPERIMENTS"]) == 1
    cfg = env["EXPERIMENTS"][0]
    assert cfg["name"] == dp.PROFILE and cfg["parser_mode"] == "code"
    assert not (set(cfg) & xp.RETIRED)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if not any(line.startswith(("!","%")) for line in source.splitlines()):
                ast.parse(source)
