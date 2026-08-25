from __future__ import annotations


# Keep this ontology stable so results from different prompting modes remain comparable.
CONDITIONS = (
    "dental_implant",
    "prosthetic_restoration",
    "dental_filling",
    "endodontic_treatment",
    "carious_lesion",
    "periodontal_bone_loss",
    "impacted_tooth",
    "periapical_lesion",
    "root_fragment",
    "furcation_lesion",
    "apical_surgery",
    "root_resorption",
    "orthodontic_device",
    "surgical_device",
)


CONDITION_LABELS = {
    "dental_implant": "dental implant",
    "prosthetic_restoration": "prosthetic restoration such as a crown or bridge",
    "dental_filling": "dental filling/restoration",
    "endodontic_treatment": "root canal/endodontic treatment",
    "carious_lesion": "carious lesion",
    "periodontal_bone_loss": "periodontal/alveolar bone loss",
    "impacted_tooth": "impacted or unerupted tooth",
    "periapical_lesion": "periapical radiolucent lesion / apical inflammatory lesion",
    "root_fragment": "residual root or root fragment",
    "furcation_lesion": "furcation bone-loss lesion",
    "apical_surgery": "evidence of apical surgery/apicoectomy",
    "root_resorption": "internal or external root resorption",
    "orthodontic_device": "orthodontic appliance/device",
    "surgical_device": "other visible surgical fixation/device",
}


STATUS_INSTRUCTION = """Use only evidence visible on this radiograph. Do not infer a finding merely because another finding is present.
If image quality or visibility is insufficient, choose UNCERTAIN rather than guessing.
Reason briefly, then end with exactly one of:
<answer>PRESENT</answer>
<answer>ABSENT</answer>
<answer>UNCERTAIN</answer>"""


def _record(question_id: str, layer: str, target: str, question: str, max_tokens: int) -> dict:
    return {
        "question_id": question_id,
        "layer": layer,
        "target": target,
        "question": question.strip(),
        "max_tokens": max_tokens,
    }


def broad_records() -> list[dict]:
    """Four independent broad passes. BASIC mode intentionally stops here."""
    return [
        _record(
            "broad_overall",
            "BROAD",
            "overall",
            """Examine this panoramic dental radiograph as a dental radiology screening image.
List the radiographically visible abnormalities and prior dental treatments that are reasonably supported by the image. Also mention important regions that are not reliably assessable. Do not invent clinical history.
Give a concise reasoning section and a concise final answer inside <answer>...</answer>.""",
            1024,
        ),
        _record(
            "broad_teeth_restorations",
            "BROAD",
            "teeth_and_restorations",
            """Inspect the teeth and dental restorations across the entire panoramic radiograph.
Look broadly for missing/defective tooth structure, residual roots, impacted teeth, fillings, crowns/bridges, implants, root-canal-treated teeth, orthodontic appliances, and other obvious treatment-related findings.
State only findings supported by the image and give the final concise list inside <answer>...</answer>.""",
            1024,
        ),
        _record(
            "broad_periapical_periodontal",
            "BROAD",
            "periapical_and_periodontal",
            """Inspect the periodontal, alveolar-bone, root, furcation, and periapical regions throughout this panoramic radiograph.
Report visible abnormalities and where they are approximately located. If a region cannot be assessed reliably, say so.
Give the final concise findings inside <answer>...</answer>.""",
            1024,
        ),
        _record(
            "broad_jaw_position_devices",
            "BROAD",
            "jaw_position_and_devices",
            """Inspect this panoramic radiograph for jaw-bone abnormalities, eruption/position abnormalities, impacted teeth, root fragments, and visible orthodontic or surgical devices.
Report only radiographically supported findings and their approximate locations. Put the concise final findings inside <answer>...</answer>.""",
            1024,
        ),
    ]


def family_records() -> list[dict]:
    """Intermediate disease-family questions used before atomic screening."""
    families = [
        (
            "family_restorative",
            "restorative_and_implant",
            "restorations and treatment-related findings: fillings, crowns/bridges, dental implants, root-canal treatment, orthodontic devices, and surgical devices",
        ),
        (
            "family_periodontal",
            "periodontal",
            "periodontal and supporting-bone findings, especially alveolar bone loss and furcation involvement",
        ),
        (
            "family_periapical_root",
            "periapical_and_root",
            "periapical and root abnormalities, including periapical lesions, residual roots/root fragments, root resorption, and evidence of apical surgery",
        ),
        (
            "family_tooth_pathology_position",
            "tooth_pathology_and_position",
            "tooth pathology and position findings, especially carious lesions and impacted/unerupted teeth",
        ),
    ]
    return [
        _record(
            qid,
            "FAMILY",
            target,
            f"""Evaluate this panoramic radiograph specifically for {description}.
List the findings that are visible, the main evidence, and approximate region. Do not force a diagnosis when visibility is inadequate.
Put the concise final result inside <answer>...</answer>.""",
            768,
        )
        for qid, target, description in families
    ]


def atomic_records() -> list[dict]:
    """One independent classification question per ontology condition."""
    records: list[dict] = []
    for condition in CONDITIONS:
        label = CONDITION_LABELS[condition]
        records.append(
            _record(
                f"atomic_{condition}",
                "ATOMIC_FINDING",
                condition,
                f"""Evaluate one finding only: {label}.
Across the whole panoramic radiograph, is this finding radiographically present?
{STATUS_INSTRUCTION}""",
                512,
            )
        )
    return records


def location_records(condition: str) -> list[dict]:
    """Coarse-to-fine text localization for a candidate condition.

    Unknown is explicitly allowed at every level; this is important because DentalGPT is
    a VLM, not a pixel-level detector/segmenter.
    """
    if condition not in CONDITION_LABELS:
        raise KeyError(f"Unknown condition: {condition}")
    label = CONDITION_LABELS[condition]

    return [
        _record(
            f"loc_{condition}_coarse",
            "LOCATION_COARSE",
            condition,
            f"""Focus only on the suspected {label}.
If it is visible, localize every visible site at the coarsest reliable level: maxilla or mandible; patient left, patient right, midline/bilateral; anterior or posterior.
If the finding itself is not reliably visible, say UNKNOWN rather than inventing a location.
Put only the concise location result inside <answer>...</answer>.""",
            512,
        ),
        _record(
            f"loc_{condition}_anatomic",
            "LOCATION_ANATOMIC",
            condition,
            f"""Focus only on the suspected {label}.
For each reliably visible site, describe the anatomical relation as specifically as the panoramic image permits: tooth/crown/root, root apex/periapical region, alveolar crest, furcation, edentulous ridge, or jawbone region.
If this cannot be determined reliably, answer UNKNOWN.
Put the concise result inside <answer>...</answer>.""",
            512,
        ),
        _record(
            f"loc_{condition}_tooth",
            "LOCATION_TOOTH",
            condition,
            f"""Focus only on the suspected {label}.
Give the quadrant and FDI tooth number(s) only when they can be identified reliably from this panoramic radiograph. If exact tooth numbering is uncertain, give the coarser region and explicitly mark the FDI number as UNKNOWN.
Do not guess. Put the concise result inside <answer>...</answer>.""",
            512,
        ),
    ]


ORCHESTRATOR_SYSTEM_PROMPT = """You are a conservative text-only aggregation component for an experimental dental-radiograph pipeline.
You NEVER see the radiograph. You may use only the supplied DentalGPT observations.

Rules:
1. Return every condition in the closed ontology exactly once.
2. Never invent visual evidence, a tooth number, a location, or a diagnosis that is absent from the observations.
3. When deterministic_atomic_statuses contains a condition, copy that status exactly. Do not upgrade or downgrade it.
4. In BASIC mode, where atomic statuses are unavailable, infer cautiously from broad observations; use UNCERTAIN whenever support is ambiguous.
5. Broad/family observations are supporting context and conflict checks. They must not override an available deterministic atomic status.
6. Populate location only from location-layer observations for that same condition. Unknown or unsupported fields must be null.
7. If observations conflict, describe the conflict in the conflict field; do not silently resolve it by inventing certainty.
8. Evidence must be a concise paraphrase of supplied observations, not new radiographic interpretation.
9. The report must clearly state that this is experimental model output and not a clinical diagnosis.
"""
