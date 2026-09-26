"""Dataset vocabulary and projections. These labels never determine model questions."""
from copy import deepcopy

UMFIH_CLASSES = ("dental_implant", "prosthetic_restoration", "dental_filling", "endodontic_treatment",
                 "carious_lesion", "periodontal_bone_loss", "impacted_tooth", "periapical_lesion",
                 "root_fragment", "furcation_lesion", "apical_surgery", "root_resorption",
                 "orthodontic_device", "surgical_device")
CONDITION_TASKS = {
    "dental_implant": ("implant",), "prosthetic_restoration": ("prosthetic_crown", "prosthetic_bridge"),
    "dental_filling": ("fillings",), "endodontic_treatment": ("root_canal_therapy",),
    "carious_lesion": ("caries",), "impacted_tooth": ("impacted_tooth",),
    "periapical_lesion": ("apical_periodontitis",), "root_fragment": ("residual_root",),
}
PROXY_CONDITIONS = {"periapical_lesion"}
TRAINED = tuple(CONDITION_TASKS)
PRIMARY_CONDITIONS = tuple(c for c in TRAINED if c not in PROXY_CONDITIONS)
LABELS = {c: c.replace("_", " ").capitalize() for c in UMFIH_CLASSES}
MAPPING_VERSION = 1


def project_result(result):
    """Copy clinical tasks into dataset groups; leave the clinical artifact unchanged."""
    import dental_pipeline as dp
    if result.get("benchmark_projection"):
        return result
    out = dict(result)
    if result.get("schema") != dp.RESULT_SCHEMA:
        # Saved unsupported predictions are retained in the original artifact only.
        out["findings"] = deepcopy(result["findings"])
    else:
        out["findings"] = {}
        tasks = result["tasks"]
        for condition, keys in CONDITION_TASKS.items():
            blocks = [tasks.get(k, {}) for k in keys]
            presence = dp._any_yes(b.get("presence") for b in blocks)
            regions = dp._merge_regions(keys, tasks, "presence", "regions")
            out["findings"][condition] = {
                "asked": True, "tasks": list(keys), "presence": presence, "whole_image": presence,
                "regions": regions, "whole_image_regions": regions, "unresolved_regions": [],
                **dp.count_block(presence, regions)}
    for condition in UMFIH_CLASSES:
        if condition not in CONDITION_TASKS:
            out["findings"][condition] = {"asked": False, "tasks": [], "presence": None,
                "whole_image": None, "regions": None, "whole_image_regions": None,
                "unresolved_regions": [], "region_count": None, "count_status": "not_assessed"}
    out["benchmark_projection"] = MAPPING_VERSION
    return out


def project_results(results):
    return {key: project_result(value) for key, value in results.items()}
