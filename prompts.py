"""Fixed question bank and orchestrator instructions."""

from __future__ import annotations


CONDITIONS = {
    "dental_implant": "dental implant",
    "prosthetic_restoration": "prosthetic restoration such as a crown or bridge",
    "dental_filling": "dental filling or obturation",
    "endodontic_treatment": "endodontic or root-canal treatment",
    "carious_lesion": "carious lesion",
    "periodontal_bone_loss": "periodontal or alveolar bone loss",
    "impacted_tooth": "impacted tooth",
    "periapical_lesion": "periapical lesion or apical periodontitis",
    "root_fragment": "root fragment or residual root",
    "furcation_lesion": "furcation lesion",
    "apical_surgery": "evidence of apical surgery",
    "root_resorption": "root resorption",
    "orthodontic_device": "orthodontic device",
    "surgical_device": "surgical device",
}


BROAD_QUESTIONS = [
    (
        "broad_complete",
        "Carefully inspect this panoramic dental radiograph. Describe all clinically "
        "important visible findings and abnormalities. Mention only image-supported "
        "findings and clearly state uncertainty.",
    ),
    (
        "broad_abnormalities",
        "What dental or jaw abnormalities may be visible in this panoramic radiograph? "
        "Briefly describe the visible evidence for each possible abnormality.",
    ),
    (
        "broad_treatments",
        "What previous dental treatments, restorations, prostheses, or devices are "
        "visible in this panoramic radiograph?",
    ),
    (
        "broad_safety_net",
        "Inspect the entire panoramic radiograph again. Is there any additional "
        "important finding that was easy to overlook? Describe it only when visibly supported.",
    ),
]


FAMILIES = {
    "tooth_roots": {
        "targets": ["carious_lesion", "root_fragment", "root_resorption"],
        "question": "Are abnormalities involving tooth structure or tooth roots visible? Describe them briefly.",
    },
    "periapical": {
        "targets": ["periapical_lesion", "apical_surgery"],
        "question": "Are abnormalities involving tooth apices or periapical regions visible?",
    },
    "supporting_bone": {
        "targets": ["periodontal_bone_loss", "furcation_lesion"],
        "question": "Are abnormalities involving alveolar bone, periodontal support, or furcation regions visible?",
    },
    "eruption_position": {
        "targets": ["impacted_tooth"],
        "question": "Are any teeth abnormally positioned, unerupted, or impacted?",
    },
    "treatments_devices": {
        "targets": [
            "dental_implant", "prosthetic_restoration", "dental_filling",
            "endodontic_treatment", "orthodontic_device", "surgical_device",
        ],
        "question": "Which dental treatments, restorations, or treatment-related devices are visible?",
    },
}


ATOMIC_TEMPLATE = """Carefully inspect this panoramic dental radiograph for {label}.

Is {label} visibly present? Briefly describe only the image evidence.
If the image is insufficient or ambiguous, say so.

Finish with exactly one of:
<answer>PRESENT</answer>
<answer>ABSENT</answer>
<answer>UNCERTAIN</answer>"""


LOCATION_FIELDS = {
    "distribution": """For the suspected {label}, what is its distribution?
Answer with: localized, generalized, multiple separated areas, or uncertain.""",
    "arch": """For the suspected {label}, which arch is involved?
Answer with: maxilla, mandible, both arches, or uncertain.""",
    "side_area": """For the suspected {label}, give the patient's anatomical side and area.
Use: right, left, bilateral, or midline; and anterior, posterior, both, or uncertain.""",
    "relation": """For the suspected {label}, what is the closest anatomical relationship?
Use: tooth crown, tooth root, root apex, alveolar crest, furcation, jawbone,
around an impacted tooth, or uncertain.""",
    "fdi_tooth": """For the suspected {label}, state the most likely FDI tooth number only
if clearly identifiable. Otherwise answer: Tooth uncertain.""",
}


ORCHESTRATOR_SYSTEM_PROMPT = """You combine observations produced by DentalGPT after it
inspected one panoramic radiograph through several questions.

Use only the supplied observations; do not inspect or diagnose the image yourself.
Group statements by canonical condition. Specific atomic observations are more useful
than vague broad statements, but meaningful disagreement must produce UNCERTAIN.
Do not infer disease from treatment: filling is not caries, root-canal treatment is not
periapical disease, implant is not active pathology, and root fragment is not resorption.
Use only location information explicitly supplied. Prefer a coarse reliable region over
an invented tooth number.

Return every condition in the closed ontology exactly once. Status must be PRESENT,
ABSENT, or UNCERTAIN. In BASIC mode, silence is UNCERTAIN rather than ABSENT.
Also write one concise dentist-facing report. This output is decision support and is not
a substitute for a dentist's interpretation."""


def broad_records() -> list[dict]:
    return [
        {"question_id": qid, "layer": "BROAD", "target": None, "question": question}
        for qid, question in BROAD_QUESTIONS
    ]


def family_records() -> list[dict]:
    return [
        {
            "question_id": f"family_{family}",
            "layer": "FAMILY",
            "family": family,
            "target": None,
            "targets": data["targets"],
            "question": data["question"],
        }
        for family, data in FAMILIES.items()
    ]


def atomic_records() -> list[dict]:
    return [
        {
            "question_id": f"atomic_{condition}",
            "layer": "ATOMIC_FINDING",
            "target": condition,
            "question": ATOMIC_TEMPLATE.format(label=label),
        }
        for condition, label in CONDITIONS.items()
    ]


def location_records(condition: str) -> list[dict]:
    label = CONDITIONS[condition]
    return [
        {
            "question_id": f"location_{field}_{condition}",
            "layer": "LOCATION",
            "target": condition,
            "location_field": field,
            "question": template.format(label=label),
        }
        for field, template in LOCATION_FIELDS.items()
    ]
