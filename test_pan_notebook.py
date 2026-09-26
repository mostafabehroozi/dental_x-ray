"""Execute configuration, saved inference, evaluation and inspection cells offline."""
import json
from pathlib import Path
from unittest.mock import patch
import dental_pipeline as dp
import dental_eval as ev
import dental_analysis as da
import experiments as xp
import llm_api
import location_adapter as la
import run_monitor as mon
from pan_test_support import Runner


def test_notebook_saved_baseline_workflow(tmp_path):
    notebook = json.loads(Path("main_notebook.ipynb").read_text(encoding="utf8"))
    scope = dict(dp=dp, ev=ev, da=da, xp=xp, llm_api=llm_api, la=la, mon=mon, Path=Path, json=json)
    with patch.object(llm_api, "secret", return_value=None), patch.object(llm_api, "configure_providers"):
        exec("".join(notebook["cells"][3]["source"]), scope)
    assert scope["CALL_LOG"] in mon.CALL_LOG_POLICIES
    scope["OUTPUT_ROOT"] = str(tmp_path)
    cfg = xp.resolve({"name": dp.PROFILE, "output_root": str(tmp_path)})
    image = tmp_path / "image.png"; image.write_bytes(b"offline PAN")
    scope["EXPERIMENTS"] = [cfg]
    scope["DATASETS"] = [{"name": "toy_umfih"}, {"name": "toy_dentex"}]
    scope["GT"] = {
        "toy_umfih": {"image": {"path": str(image), "annotated": set(ev.CONDITIONS), "boxes": []}},
        "toy_dentex": {"image": {"path": str(image), "annotated": {"carious_lesion"}, "boxes": []}}}
    runner = Runner()
    parser = xp.parser(cfg)
    scope.update(LEDGER=mon.Ledger("offline"), open_runner=lambda cfg: runner, open_parser=lambda cfg: parser)
    for index in (8, 9, 10, 11, 12, 13, 14):
        exec(compile("".join(notebook["cells"][index]["source"]), f"cell_{index}", "exec"), scope)
    # Each dataset's smoke results are resumed by the full-run cell.
    assert len(runner.requests) == 24
    assert not scope["LEDGER"].entries
    assert len(scope["REPORTS"]) == 2
    for dataset in scope["GT"]:
        evaluation = xp.run_dir(cfg, dataset) / "evaluation"
        assert (evaluation / "task_coverage.csv").is_file()
        saved = dp.load_results(xp.run_dir(cfg, dataset))["image"]
        assert saved["schema"] == dp.RESULT_SCHEMA and saved["call_count"] == 12
