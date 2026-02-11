import json
import os
import time
from tqdm import tqdm
from openai import OpenAI

# Use env OPENAI_BASE_URL, OPENAI_API_KEY (optional: OPENAI_BASE_URL for custom endpoint)
_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Set it in your environment or env.local.sh.")
        kwargs = {"api_key": api_key}
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        if base_url:
            kwargs["base_url"] = base_url
        _client = OpenAI(**kwargs)
    return _client

model_list = ["gpt-4o-mini", "gpt-4o", "gemini-2.5-flash", "gpt-5-nano"]

CUE_DIALOGUE_PROMPT = """
You are generating short conversational cues for testing memory recall capabilities.

Generate short dialogues (2 lines) based on the relation type provided.

Relation Type: {relation_type}

Relation type meanings:
- Causal: an earlier cause or condition affects a later event.
- State: a physical or emotional state influences later behavior.
- Goal: a long-term intention or plan influences current choices.
- Value: a belief or value shapes later reactions.


Examples:
- Cue: "A: Family always comes first for me.\nB: Same here, they're everything."

- Cue: "A: I volunteered to mentor a junior colleague because I wanted to give back at work.\nB: That's generous; they're lucky to have you."

- Cue: "A: The doctor said my cholesterol levels are high.\nB: You should take care of that."

- Cue: "A: I need to save money for the car repairs.\nB: Sounds like a good idea."

- Cue: "A: My last relationship taught me a lot about communication.\nB: That's important; every experience can help us grow."

Requirements:
- Create exactly 2 lines of dialogue for each example:
  - Line 1 (A): A person mentions an event, state, feeling, or decision that is MEMORABLE and RECALLABLE - something distinctive enough to be remembered weeks or months later.
  - Line 2 (B): Another person gives a short, natural reaction that CLOSES the conversation naturally. B's response should be a STATEMENT, not a question.
- Ensure dialogue closure: The conversation should feel naturally concluded, not open-ended.
- Memory anchor: A's line should contain a distinctive detail, personal revelation, or significant information that serves as a strong memory cue.
- Make it sound like realistic, natural daily conversation.
- Vary topics across work, family, relationships, health, travel, learning, etc.
- Do NOT include explanations or markdown.
- Output a valid JSON array ONLY.



Output strictly in this format:
[
  {{
    "relation_type": "{relation_type}",
    "cue_dialogue": "A: ...\\nB: ..."
  }},
  ...
]

Generate {num_samples} examples.
"""

def safe_json_parse(resp: str):
    if not resp or not isinstance(resp, str):
        return []

    try:
        if resp.startswith("'") or resp.startswith('"'):
            resp = resp.strip("'").strip('"')
        try:
            return json.loads(resp)
        except json.JSONDecodeError:
            decoded = json.loads(resp.encode().decode('unicode_escape'))
            if isinstance(decoded, str):
                return json.loads(decoded)
            return decoded
    except Exception as e:
        print(f"safe_json_parse error: {e}")
        return []

def call_openai(prompt: str, model="gpt-4o-mini", temperature=1.0):
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API call failed: {e}")
        return ""

if __name__ == "__main__":
    output_file = "cue_dialogue_data.json"
    
    existing_data = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            print(f"Found {len(existing_data)} existing data entries, will append new data")
        except Exception as e:
            print(f"Error reading existing file: {e}")
            existing_data = []
    
    all_new_data = []
    
    for model in model_list:
        print(f"\n\n==============================")
        print(f"Generating cue_dialogue with model: {model}")
        print(f"==============================")
        
        new_data = []
        relation_types = ["causal", "state", "goal", "value"]
        
        for relation in relation_types:
            print(f"\n=== {model} | Generating cue_dialogues for relation_type: {relation} ===")
            prompt = CUE_DIALOGUE_PROMPT.format(
                relation_type=relation,
                num_samples=13
            )

            for i in tqdm(range(1)):
                resp = call_openai(prompt, model=model, temperature=0.9)
                if not resp:
                    continue
                    
                js = safe_json_parse(resp)
                if not js:
                    continue
                    
                for j in js:
                    j["relation_type"] = relation
                    j["model_name"] = model
                    new_data.append(j)
                    
                time.sleep(0.5)
        
        all_new_data.extend(new_data)
        print(f" {model} generation complete: {len(new_data)} new entries")

    all_data = existing_data + all_new_data
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n All data saved to unified file → {output_file}")
    print(f" Total data entries: {len(all_data)}")
    print(f" New entries this run: {len(all_new_data)}")