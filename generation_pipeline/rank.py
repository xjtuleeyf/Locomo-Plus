import json
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util


def load_all_data():
    # RANK_INPUT: path to complete cue-query JSON (default: complete_data_all_models.json in script dir)
    default_path = Path(__file__).resolve().parent / "complete_data_all_models.json"
    input_file = os.environ.get("RANK_INPUT", str(default_path))

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"Data loaded: {len(data)} entries")
            return data
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        return []
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return []

def calculate_similarity_scores(data):
    print("Loading similarity models...")
    
    bm25_corpus = [d["cue_dialogue"] for d in data]
    bm25_tokens = [doc.lower().split() for doc in bm25_corpus]
    bm25 = BM25Okapi(bm25_tokens)
    
    # Optional: set SENTENCE_TRANSFORMER_MPNET and SENTENCE_TRANSFORMER_BGE to local paths
    mpnet_path = os.environ.get("SENTENCE_TRANSFORMER_MPNET", "sentence-transformers/multi-qa-mpnet-base-dot-v1")
    bge_path = os.environ.get("SENTENCE_TRANSFORMER_BGE", "BAAI/bge-small-en-v1.5")
    mpnet = SentenceTransformer(mpnet_path)
    bge = SentenceTransformer(bge_path)
    
    print("Encoding embeddings...")
    
    cue_texts = [d["cue_dialogue"] for d in data]
    query_texts = [d["trigger_query"] for d in data]
    
    cue_emb_mpnet = mpnet.encode(cue_texts, batch_size=64, convert_to_tensor=True, show_progress_bar=True)
    query_emb_mpnet = mpnet.encode(query_texts, batch_size=64, convert_to_tensor=True, show_progress_bar=True)
    
    cue_emb_bge = bge.encode(cue_texts, batch_size=64, convert_to_tensor=True, show_progress_bar=True)
    query_emb_bge = bge.encode(query_texts, batch_size=64, convert_to_tensor=True, show_progress_bar=True)
    
    print("Calculating similarity scores...")
    results = []
    
    all_mpnet_scores = []
    all_bge_scores = []
    all_bm25_scores = []
    all_combined_scores = []
    
    for i, d in enumerate(tqdm(data)):
        cue = d["cue_dialogue"]
        query = d["trigger_query"]
        
        bm25_score = bm25.get_scores(query.lower().split())[i]
        
        mpnet_score = float(util.cos_sim(cue_emb_mpnet[i], query_emb_mpnet[i]))
        bge_score = float(util.cos_sim(cue_emb_bge[i], query_emb_bge[i]))
        
        combined = np.mean([mpnet_score, bge_score]) * 0.8 + (bm25_score / (bm25_score + 1)) * 0.2
        
        all_mpnet_scores.append(mpnet_score)
        all_bge_scores.append(bge_score)
        all_bm25_scores.append(bm25_score)
        all_combined_scores.append(combined)
        
        result_item = d.copy()
        result_item.update({
            "scores": {
                "mpnet": mpnet_score,
                "bge": bge_score,
                "bm25": bm25_score,
                "combined": combined
            }
        })
        results.append(result_item)
    
    print("Calculating ranks...")
    mpnet_ranks = calculate_ranks(all_mpnet_scores)
    bge_ranks = calculate_ranks(all_bge_scores)
    bm25_ranks = calculate_ranks(all_bm25_scores)
    combined_ranks = calculate_ranks(all_combined_scores)
    
    for i, result_item in enumerate(results):
        result_item["ranks"] = {
            "mpnet": mpnet_ranks[i],
            "bge": bge_ranks[i],
            "bm25": bm25_ranks[i],
            "combined": combined_ranks[i]
        }
        result_item["final_similarity_score"] = result_item["scores"]["combined"]
    
    return results

def calculate_ranks(scores):
    sorted_indices = np.argsort(scores)
    ranks = np.zeros(len(scores), dtype=int)
    for rank, idx in enumerate(sorted_indices, 1):
        ranks[idx] = rank
    return ranks.tolist()

if __name__ == "__main__":
    print("Loading data...")
    all_data = load_all_data()
    print(f"Total data entries: {len(all_data)}")
    
    if not all_data:
        print("No data to process")
        exit()
    
    print("\nCalculating similarity scores and ranks...")
    scored_data = calculate_similarity_scores(all_data)
    
    print("\nSorting by combined similarity (low similarity first)...")
    sorted_by_combined = sorted(scored_data, key=lambda x: x["final_similarity_score"])
    
    output_file = "evaluated_similarity_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sorted_by_combined, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved: {output_file}")
    
    print("\nTop 10 low similarity samples:")
    print("=" * 100)
    for i, item in enumerate(sorted_by_combined[:10]):
        print(f"\n#{i+1} | Combined similarity: {item['final_similarity_score']:.4f} (Rank: {item['ranks']['combined']})")
        print(f"Type: {item['relation_type']} | Model: {item['model_name']}")
        print(f"MPNet: {item['scores']['mpnet']:.4f} (Rank: {item['ranks']['mpnet']})")
        print(f"BGE: {item['scores']['bge']:.4f} (Rank: {item['ranks']['bge']})") 
        print(f"BM25: {item['scores']['bm25']:.4f} (Rank: {item['ranks']['bm25']})")
        print(f"Cue: {item['cue_dialogue']}")
        print(f"Query: {item['trigger_query']}")
        print("-" * 100)
    
    print("\nStatistics:")
    combined_scores = [item["final_similarity_score"] for item in sorted_by_combined]
    print(f"Combined similarity range: {min(combined_scores):.4f} - {max(combined_scores):.4f}")
    print(f"Average combined similarity: {np.mean(combined_scores):.4f}")
    print(f"Median combined similarity: {np.median(combined_scores):.4f}")
    print(f"Standard deviation: {np.std(combined_scores):.4f}")
    
    print("\nStatistics by model:")
    models = set(item['model_name'] for item in sorted_by_combined)
    for model in models:
        model_scores = [item["final_similarity_score"] for item in sorted_by_combined if item['model_name'] == model]
        print(f"{model}: {len(model_scores)} entries, average similarity: {np.mean(model_scores):.4f}")
    
    print("\nStatistics by relation type:")
    relation_types = set(item['relation_type'] for item in sorted_by_combined)
    for rel_type in relation_types:
        rel_scores = [item["final_similarity_score"] for item in sorted_by_combined if item['relation_type'] == rel_type]
        print(f"{rel_type}: {len(rel_scores)} entries, average similarity: {np.mean(rel_scores):.4f}")