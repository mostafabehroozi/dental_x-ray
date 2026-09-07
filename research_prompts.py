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
FDM_COMBINED_PROMPT_TEMPLATES = {
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

# Provider-neutral API prompts: final outputs only, without a model-specific
# reasoning transcript. Task units and response contracts stay fixed across wording variants.
API_VISUAL_RULES = (
    "Use only evidence visible in the supplied dental radiograph. "
    "Evaluate each requested finding independently; one treatment or device is not proof of another finding. "
    "Count each supported instance once using the specified counting unit, even if it has several visible components. "
    "Do not infer hidden instances, clinical history, or diagnoses outside the requested categories. "
    "Do not count equivocal artifacts or overlapping anatomy as confirmed findings. "
    "Presence anywhere in the requested scope is sufficient; it need not be generalized throughout that scope."
)
API_ATOMIC_TASK = (
    "\nTarget finding: {finding}\nScope: {image_scope}\nCounting unit: {count_subject}\n\n"
    + API_VISUAL_RULES +
    " Inspect the entire requested scope, including visible anterior and posterior areas. "
    "Use surrounding anatomy for context, but count only instances inside the requested scope."
)
API_COMBINED_FORMAT = (
    '\nReturn exactly one <answer> tag containing a JSON object with only "choice" and "count". '
    '"count" is the number of visibly supported instances, as a non-negative integer. '
    'Use "choice":"A" when count is greater than zero; use "choice":"B" when count is zero. '
    'Do not include explanations, Markdown fences, extra keys, or reasoning text. '
    'Format examples only (not evidence about this image): '
    '<answer>{{"choice":"A","count":3}}</answer> or <answer>{{"choice":"B","count":0}}</answer>.'
)
API_ATOMIC_OPENINGS = {
    "atomic_1": "Assess the specified finding in the specified scope of this dental radiograph.",
    "atomic_2": "Inspect this dental radiograph for the single target finding within the requested scope.",
    "atomic_3": "Perform a focused visual assessment of the target finding, limited to the scope below.",
}
LLM_ATOMIC_PROMPT_TEMPLATES = {
    key: opening + API_ATOMIC_TASK + API_COMBINED_FORMAT
    for key, opening in API_ATOMIC_OPENINGS.items()
}
API_PRESENCE_PROMPT_TEMPLATES = {
    key: opening + "\nTarget finding: {finding}\nScope: {image_scope}\n\n" + API_VISUAL_RULES +
         "\nA. True: at least one visibly supported instance.\nB. False: no visibly supported instance.\n"
         "Return only <answer>A</answer> or <answer>B</answer>, without explanations or reasoning text."
    for key, opening in API_ATOMIC_OPENINGS.items()
}
API_COUNT_PROMPT_TEMPLATES = {
    key: opening + API_ATOMIC_TASK +
         "\nHow many supported instances are visible? Return only the non-negative integer "
         "inside one <answer>...</answer> tag, without explanations or reasoning text."
    for key, opening in API_ATOMIC_OPENINGS.items()
}
API_BROAD_PROMPT_TEMPLATES = {
    key: opening + "\n" + API_VISUAL_RULES +
         " Inspect both jaws, including anterior and posterior areas, for all listed categories. "
         "Report whole-image totals using the same counting units for every category. "
         "Keep the final response concise; do not include a reasoning transcript or treatment recommendations."
    for key, opening in {
        "broad_1": "Analyze the complete dental radiograph and report all findings in the categories listed below.",
        "broad_2": "Review the whole dental radiograph systematically and summarize every listed finding category.",
        "broad_3": "Produce one complete image-based assessment covering all of the following finding categories.",
    }.items()
}
API_NARRATIVE_FORMAT = (
    "\nWrite one overall report in natural language: a short image-quality/limitations summary, "
    "then a table with exactly one row per listed category, in the listed order. "
    "Columns: category key | status (PRESENT, ABSENT, or UNCERTAIN) | whole-image count | short visible evidence or limitation. "
    "Use an integer when an exact count is supported. For an uncountable positive finding use PRESENT and UNKNOWN; "
    "for uncertain presence use UNCERTAIN and UNKNOWN. Use ABSENT and 0 only when absence is assessable. "
    "If the image cannot be assessed, explicitly mark affected categories UNCERTAIN with UNKNOWN counts. "
    "Keep table counts consistent with the prose and count each instance once. "
    "Do not output JSON, a reasoning transcript, or invented tooth numbers."
)
ADAPTER_PROMPT_TEMPLATES = {
    "adapter_1": "Convert the supplied analyzer report into finding counts using only that report.",
    "adapter_2": "Extract the reported counts for each listed finding. Use no evidence beyond the supplied report.",
    "adapter_3": "Normalize the supplied report to the required count schema without adding clinical findings.",
}

PROMPT_TEMPLATES = {
    "presence": FDM_PRESENCE_PROMPT_TEMPLATES,
    "count": FDM_COUNT_PROMPT_TEMPLATES,
    "combined": FDM_COMBINED_PROMPT_TEMPLATES,
    "broad": BROAD_PROMPT_TEMPLATES,
    "adapter": ADAPTER_PROMPT_TEMPLATES,
}
API_PROMPT_TEMPLATES = {
    "presence": API_PRESENCE_PROMPT_TEMPLATES,
    "count": API_COUNT_PROMPT_TEMPLATES,
    "combined": LLM_ATOMIC_PROMPT_TEMPLATES,
    "broad": API_BROAD_PROMPT_TEMPLATES,
}
DEFAULT_STRATEGIES = {
    "broad_whole": {"mode": "broad", "location_mode": "whole", "finding_group": "all_14", "template_id": "broad_1"},
    **{
        "atomic_" + level: {"mode": "atomic", "location_mode": level, "finding_group": "all_14", "template_id": "atomic_1"}
        for level in LOCATION_MODES
    },
}
