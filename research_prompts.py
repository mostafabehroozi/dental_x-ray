"""Editable defaults for research only; production prompts are independent."""
from prompts import CONDITIONS, CONDITION_LABELS

# Source: DentalGPT arXiv:2512.11558v1, Section 3.2 (p.5), Figure 7 (p.10),
# Figure 9 (p.13). The paper describes the tag contract but does not publish the
# exact appended training instruction or all training templates. These are close
# adaptations. Figure 7 is a benchmark example, not a released training sample.
ATOMIC_FINDING_LABELS = {
    "dental_implant": "Dental implant",
    "prosthetic_restoration": "Prosthetic restoration",
    "dental_filling": "Dental filling",
    "endodontic_treatment": "Root canal treatment",
    "carious_lesion": "Carious lesion",
    "periodontal_bone_loss": "Periodontal/alveolar bone loss",
    "impacted_tooth": "Impacted tooth",
    "periapical_lesion": "Periapical lesion",
    "root_fragment": "Root fragment",
    "furcation_lesion": "Furcation lesion",
    "apical_surgery": "Apical surgery",
    "root_resorption": "Root resorption",
    "orthodontic_device": "Orthodontic device",
    "surgical_device": "Surgical device",
}

# Reconstructed from Section 3.2; not quoted as the undisclosed fixed suffix.
REASONING_FORMAT = "Put your reasoning inside <think>...</think> and your final answer inside <answer>...</answer>."
PRESENCE_FORMAT = REASONING_FORMAT + " The final answer must be only A or B."
# Figure 9 has a prose final answer. A single integer is our evaluation constraint.
COUNT_FORMAT = REASONING_FORMAT + " The final answer must be only a non-negative integer."

FDM_PRESENCE_PROMPT_TEMPLATES = {
    # Figure 7's panoramic question, with only condition/scope substituted.
    "atomic_1": "Kindly evaluate if the condition '{finding}' is present in {image_scope}.\nA. True\nB. False\n\n" + PRESENCE_FORMAT,
    # Adapted from Figure 7's intraoral classification wording.
    "atomic_2": "Analyze {image_scope} to assess whether the condition '{finding}' is present.\nA. True\nB. False\n\n" + PRESENCE_FORMAT,
    # Our fallback paraphrase, not an additional template published by the paper.
    "atomic_3": "Is the condition '{finding}' present in {image_scope}?\nA. True\nB. False\n\n" + PRESENCE_FORMAT,
}

FDM_COUNT_PROMPT_TEMPLATES = {
    # The filling question follows Figure 9; other categories use a direct How many question.
    "atomic_1": "{count_question}\n\n" + COUNT_FORMAT,
    "atomic_2": "How many {count_subject} can be identified in {count_scope}?\n\n" + COUNT_FORMAT,
    "atomic_3": "What is the number of {count_subject} visible in {count_scope}?\n\n" + COUNT_FORMAT,
}

# Combined presence/count JSON is a research adaptation, not DentalGPT's reported
# multiple-choice training contract. Use presence_then_count for the closer protocol.
COMBINED_FORMAT = (
    "If A is correct, count {count_subject}. If B is correct, the count is 0. "
    + REASONING_FORMAT +
    ' The final answer must be a JSON object with "choice" ("A" or "B") and '
    '"count" (a non-negative integer). For A, count must be positive; for B, count must be 0.'
)
LLM_ATOMIC_PROMPT_TEMPLATES = {
    key: text.replace(PRESENCE_FORMAT, COMBINED_FORMAT)
    for key, text in FDM_PRESENCE_PROMPT_TEMPLATES.items()
}

COUNT_SUBJECTS = {
    "dental_implant": "distinct dental implants",
    "prosthetic_restoration": "distinct prosthetic restorations",
    "dental_filling": "teeth with visible dental fillings",
    "endodontic_treatment": "teeth showing endodontic treatment",
    "carious_lesion": "distinct carious lesions",
    "periodontal_bone_loss": "distinct regions of periodontal or alveolar bone loss",
    "impacted_tooth": "impacted or unerupted teeth",
    "periapical_lesion": "distinct periapical lesions",
    "root_fragment": "distinct residual roots or root fragments",
    "furcation_lesion": "distinct involved furcation sites",
    "apical_surgery": "teeth or sites showing apical surgery",
    "root_resorption": "teeth or roots showing root resorption",
    "orthodontic_device": "distinct orthodontic devices or appliances",
    "surgical_device": "distinct surgical fixation devices",
}

FINDING_GROUPS = {
    "all_14": list(CONDITIONS),
    "disease_and_pathology": [
        "carious_lesion",
        "periodontal_bone_loss",
        "impacted_tooth",
        "periapical_lesion",
        "root_fragment",
        "furcation_lesion",
        "root_resorption",
    ],
    "treatment_and_devices": [
        "dental_implant",
        "prosthetic_restoration",
        "dental_filling",
        "endodontic_treatment",
        "apical_surgery",
        "orthodontic_device",
        "surgical_device",
    ],
}

LOCATION_MODES = {
    "whole": [("whole", "the full panoramic image (anywhere in the image; it need not be generalized)")],
    "arch": [
        ("maxilla", "the maxilla (upper jaw)"),
        ("mandible", "the mandible (lower jaw)"),
    ],
    "quadrant": [
        ("upper_right", "the patient's upper-right quadrant (FDI 18-11 when identifiable)"),
        ("upper_left", "the patient's upper-left quadrant (FDI 21-28 when identifiable)"),
        ("lower_left", "the patient's lower-left quadrant (FDI 38-31 when identifiable)"),
        ("lower_right", "the patient's lower-right quadrant (FDI 41-48 when identifiable)"),
    ],
    "six_zone": [
        ("upper_right_posterior", "the patient's upper-right posterior region (FDI 18-14 when identifiable)"),
        ("upper_anterior", "the maxillary anterior region (FDI 13-23 when identifiable)"),
        ("upper_left_posterior", "the patient's upper-left posterior region (FDI 24-28 when identifiable)"),
        ("lower_left_posterior", "the patient's lower-left posterior region (FDI 38-34 when identifiable)"),
        ("lower_anterior", "the mandibular anterior region (FDI 33-43 when identifiable)"),
        ("lower_right_posterior", "the patient's lower-right posterior region (FDI 44-48 when identifiable)"),
    ],
}

BROAD_PROMPT_TEMPLATES = {
    "broad_1": "Analyze the complete panoramic radiograph. Count each listed finding anywhere in the image.",
    "broad_2": "Inspect both jaws systematically and report the total number of visible instances for every listed finding.",
    "broad_3": "Review the entire radiograph for each category below, then provide its whole-image count.",
}
PROMPT_TEMPLATES = {
    "presence": FDM_PRESENCE_PROMPT_TEMPLATES,
    "count": FDM_COUNT_PROMPT_TEMPLATES,
    "combined": LLM_ATOMIC_PROMPT_TEMPLATES,
    "broad": BROAD_PROMPT_TEMPLATES,
}
DEFAULT_STRATEGIES = {
    "broad_whole": {"mode": "broad", "location_mode": "whole", "finding_group": "all_14", "template_id": "broad_1"},
    **{
        "atomic_" + level: {"mode": "atomic", "location_mode": level, "finding_group": "all_14", "template_id": "atomic_1"}
        for level in LOCATION_MODES
    },
}
