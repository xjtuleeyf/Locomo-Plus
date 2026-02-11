import json
import os
import time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# Use env OPENAI_BASE_URL, OPENAI_API_KEY
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

TRIGGER_QUERY_PROMPT = """
You are generating trigger queries that have implicit cognitive connections to given dialogues and create meaningful cognitive conflicts or contrasts with given dialogues, ensuring diverse perspectives in memory recall.

CRITICAL REQUIREMENT: Each of the five trigger queries must represent a DISTINCT COGNITIVE ANGLE of conflict or recall.  They should not feel similar or repetitive.  Aim for five truly different ways that the trigger could relate to the cue.

Given the cue dialogue below, generate FIVE DIFFERENT trigger queries that:
1. Have implicit connections that humans can easily recognize (memory recall)
2. Are semantically distant (don't reuse the same nouns/verbs from the cue)
3. Sound like natural, realistic daily conversation
4. Represent events or situations that occur at least one week or several months after the cue dialogue
5. The time gaps should be at least one week or more.
6. Trigger preferably create some conflict or contrast with the cue dialogue, or represent an active recall of the past event

Each trigger query should:
- Be spoken by the same person (A) in first person, weeks or months later
- Sound semantically unrelated but cognitively connected to the cue
- Be something that humans can easily recall the cue from, but a similarity-based retriever likely cannot
- Be a statement, feeling, question, or reflection — vary the types
- Avoid reusing the same nouns or verbs from the cue dialogue
- Occur at least one week or several months after the cue dialogue, creating temporal distance
- Represent a distinct angle from the other four triggers

AVOID:
- Similar-sounding triggers that feel repetitive
- Superficial connections without meaningful conflict
- Forced or unnatural contrasts
- Repeating the same type of relationship across multiple triggers

Requirements:
- Generate FIVE DISTINCT trigger queries with different contexts and time gaps
- Make them sound like realistic, natural daily conversation
- Vary topics across work, family, relationships, health, travel, learning, etc.
- Vary time gaps appropriately (one week to several months)
- Do NOT include explanations or markdown
- Include time_gap description for each (e.g., "one week later", "several months later", "a year after")
- Output a valid JSON array ONLY

EXAMPLES OF GOOD TRIGGER QUERIES:

Example 1 :
- Cue: "A: Family always comes first for me.\nB: Same here, they're everything."
- Trigger: "A: I'm considering taking that job overseas, even though it means being away from everyone."
- Time Gap: "three months later"

Example 2 :
- Cue: "A: I'm training for a marathon next month.\nB: That's impressive dedication."
- Trigger: "A: I haven't run in weeks, but I don't really miss it anymore."
- Time Gap: "two months later" 

Example 3 :
- Cue: "A: I'm so anxious about this job interview tomorrow.\nB: You'll do great, just be yourself."
- Trigger: "A: Getting laid off from that company was the best thing that ever happened to me."
- Time Gap: "six months later"
- Why it works: Transforms anxiety into positive reflection through time

Example 4 :
- Cue: "A: I'd never get a tattoo—they're so permanent.\nB: I feel the same way."
- Trigger: "A: This tattoo represents everything I've been through this year."
- Time Gap: "a year later"

Cue Dialogue:
{cue_dialogue}

Relation Type: {relation_type}

Output strictly in this format:
[
  {{
    "relation_type": "{relation_type}",
    "cue_dialogue": "{cue_dialogue}",
    "trigger_query": "A: ...",
    "time_gap": "description of time interval"
  }},
  {{
    "relation_type": "{relation_type}",
    "cue_dialogue": "{cue_dialogue}",
    "trigger_query": "A: ...",
    "time_gap": "description of time interval"
  }},
  ... (exactly five items, each with a distinct cognitive angle)
]
"""

def safe_json_parse(resp: str):
    if not resp or not isinstance(resp, str):
        return []

    resp = resp.strip()
    
    try:
        return json.loads(resp)
    except json.JSONDecodeError as e:
        print(f"First parse failed: {e}")
    
    try:
        start_idx = resp.find('[')
        end_idx = resp.rfind(']') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = resp[start_idx:end_idx]
            return json.loads(json_str)
    except Exception as e:
        print(f"JSON extraction failed: {e}")
    
    try:
        if resp.startswith("'") or resp.startswith('"'):
            resp = resp.strip("'").strip('"')
        decoded = json.loads(resp.encode().decode('unicode_escape'))
        if isinstance(decoded, str):
            return json.loads(decoded)
        return decoded
    except Exception as e:
        print(f"safe_json_parse failed: {e}")
        print(f"Original response: {resp}")
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
        print(f" API call failed: {e}")
        return ""

def load_cue_dialogues():
    # CUE_QUERY_INPUT: path to filtered cue dialogues JSON (default: selected_cue_query.json in script dir)
    default_path = Path(__file__).resolve().parent / "selected_cue_query.json"
    input_file = os.environ.get("CUE_QUERY_INPUT", str(default_path))

    if not os.path.exists(input_file):
        print(f" File does not exist: {input_file}")
        return []
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f" File loading failed: {e}")
        return []

if __name__ == "__main__":
    cue_data = load_cue_dialogues()
    if not cue_data:
        print(" No cue_dialogue data found")
        exit(1)
    
    print(f" Loaded {len(cue_data)} cue_dialogue records")
    
    output_file = "complete_data_all_models.json"
    
    existing_complete_data = []
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_complete_data = json.load(f)
            print(f" Existing {len(existing_complete_data)} complete records")
        except Exception as e:
            print(f" Failed to read existing file: {e}")
            existing_complete_data = []
    
    existing_cue_model_pairs = set()
    for item in existing_complete_data:
        key = f"{item['cue_dialogue']}_{item['model_name']}"
        existing_cue_model_pairs.add(key)
    
    all_new_data = []
    
    new_cue_data = []
    for cue_item in cue_data:
        if cue_item.get("model_name") not in model_list:
            print(f" Skipping unsupported model: {cue_item.get('model_name')}")
            continue
            
        key = f"{cue_item['cue_dialogue']}_{cue_item['model_name']}"
        if key not in existing_cue_model_pairs:
            new_cue_data.append(cue_item)
    
    print(f" Need to generate trigger_queries for {len(new_cue_data)} cue_dialogues")
    
    model_groups = {}
    for item in new_cue_data:
        model = item["model_name"]
        if model not in model_groups:
            model_groups[model] = []
        model_groups[model].append(item)
    
    print(f"\n Number of records to process per model:")
    for model, items in model_groups.items():
        print(f"   {model}: {len(items)} records")
    
    for cue_item in tqdm(new_cue_data, desc="Generating trigger_queries"):
        model = cue_item["model_name"]
        
        escaped_cue = cue_item["cue_dialogue"].replace('"', '\\"').replace('\n', '\\n')
        
        prompt = TRIGGER_QUERY_PROMPT.format(
            cue_dialogue=escaped_cue,
            relation_type=cue_item["relation_type"]
        )
        
        resp = call_openai(prompt, model=model, temperature=0.9)
        if not resp:
            print(f" Model {model} returned empty response")
            continue
        
        results = safe_json_parse(resp)
        if not results:
            print(f" JSON parsing failed, skipping this record")
            continue
            
        if not isinstance(results, list):
            results = [results]
        
        valid_count = 0
        for result in results:
            if isinstance(result, dict) and "trigger_query" in result and "time_gap" in result:
                result["model_name"] = model
                all_new_data.append(result)
                valid_count += 1
                if valid_count <= 2:
                    print(f" {model} generated: {result['trigger_query'][:50]}... (Time gap: {result['time_gap']})")
            else:
                print(f" Invalid result format: {result}")
        
        if valid_count > 0:
            print(f" {model} generated {valid_count}/5 trigger_queries for cue_dialogue")
        else:
            print(f" {model} failed to generate any valid trigger_queries")
        
        time.sleep(0.5)

    all_complete_data = existing_complete_data + all_new_data
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_complete_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n Complete data for all models saved → {output_file}")
    print(f" Total complete records: {len(all_complete_data)}")
    print(f" Newly added: {len(all_new_data)}")
    
    model_counts = {}
    for item in all_complete_data:
        model = item["model_name"]
        model_counts[model] = model_counts.get(model, 0) + 1
    
    print(f"\n Data distribution per model:")
    for model, count in model_counts.items():
        print(f"   {model}: {count} records")