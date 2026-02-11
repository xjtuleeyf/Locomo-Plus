"""
Judge prompts for six categories: multi-hop, temporal, common-sense, single-hop, adversarial, Cognitive.
Used by LLM-as-a-judge (v2) for evaluation.
"""

CONV_START_PROMPT = (
    "Below is a conversation between two people: {} and {}. "
    "The conversation takes place over multiple days, "
    "and the date of each conversation is written at the beginning of the conversation.\n\n"
)


PROMPT_TEMPLATES = {
    "multi-hop": """
You are a Fact-Checking Judge.
Your task: Compare the model's prediction with the reference answer (multi-hop fact QA).

Labels:
- "correct": The answer matches the reference entities (names, places, times) exactly.
- "partial": The answer misses some details or contains minor inaccuracies but gets the main entity right.
- "wrong": The answer is factually incorrect or hallucinates details not in the reference.

Reference Answer:
{gold}

Model Prediction:
{pred}

Relevant Evidence:
{evidence}

Return your judgment strictly in JSON format:
{{"label": "correct"|"partial"|"wrong", "reason": "<short explanation>"}}
""",

    "single-hop": """
You are a Fact-Checking Judge.
Your task: Compare the model's prediction with the reference answer (single-hop fact QA).

Labels:
- "correct": The answer matches the reference entities exactly.
- "partial": The answer misses some details but gets the main entity right.
- "wrong": The answer is factually incorrect or hallucinates details not in the reference.

Reference Answer:
{gold}

Model Prediction:
{pred}

Relevant Evidence:
{evidence}

Return your judgment strictly in JSON format:
{{"label": "correct"|"partial"|"wrong", "reason": "<short explanation>"}}
""",

    "temporal": """
You are a Temporal Logic Judge.
Your task: Check the calculation, duration, or sequence of events.

Labels:
- "correct": The calculated time, duration, or date matches the reference exactly (semantic equivalents are allowed).
- "wrong": The calculation is incorrect, the sequence is reversed, or the specific time is wrong.

Reference Answer:
{gold}

Model Prediction:
{pred}

Relevant Evidence:
{evidence}

Return your judgment strictly in JSON format:
{{"label": "correct"|"wrong", "reason": "<short explanation>"}}
""",

    "common-sense": """
You are a Knowledge Logic Judge.
Your task: Assess if the prediction applies correct commonsense/world knowledge consistent with the reference.

Labels:
- "correct": The logic and inference are sound and match the reference conclusion.
- "partial": The reasoning is mostly correct but the final conclusion is vague or slightly off.
- "wrong": The reasoning contradicts commonsense or the reference.

Reference Answer:
{gold}

Model Prediction:
{pred}

Relevant Evidence:
{evidence}

Return your judgment strictly in JSON format:
{{"label": "correct"|"partial"|"wrong", "reason": "<short explanation>"}}
""",

    "adversarial": """
You are a Skeptical Judge evaluating robustness.
The question is inherently misleading (e.g., asks about something not in the conversation).
Your task: Judge whether the model's answer conveys that "this was not mentioned in the conversation" (or equivalent refusal).

Labels:
- "correct": The prediction clearly conveys that the information was not mentioned / cannot be answered from the conversation. Score it.
- "wrong": The prediction does NOT convey that meaning—e.g., it gives a concrete answer or does not refuse. Do not score.


Model Prediction:
{pred}

Return your judgment strictly in JSON format:
{{"label": "correct"|"wrong", "reason": "<short explanation>"}}
""",

    "Cognitive": """
You are a Memory Awareness Judge.
Your task: Judge whether the Model Prediction considers or is linked to the Evidence. If there is a clear connection, the answer is correct (score 1); if not, it is wrong (no score).

Labels:
- "correct": The prediction explicitly or implicitly reflects/uses the evidence (memory or constraint). Give 1 point.
- "wrong": The prediction does not show such a link to the evidence. No point.

Memory/Evidence:
{evidence}

Model Prediction:
{pred}

Return your judgment strictly in JSON format:
{{"label": "correct"|"wrong", "reason": "<Does the prediction relate to the evidence?>"}}
""",

    "default": """
You are an expert evaluator.
Your task: Compare the prediction with the reference.

Labels:
- "correct": Factually consistent with the reference.
- "partial": Contains correct info but is incomplete.
- "wrong": Factually incorrect.

Reference Answer:
{gold}

Model Prediction:
{pred}

Relevant Evidence:
{evidence}

Return your judgment strictly in JSON format:
{{"label": "correct"|"partial"|"wrong", "reason": "<short explanation>"}}
""",
}
