import json
import hashlib
import os
from transformers import AutoTokenizer

# --- CONFIGURATION ---
INPUT_FILE = "output/all_training_data.jsonl"
CLEAN_FILE = "output/rabit0_final_cleaned.jsonl"
DUPLICATE_FILE = "output/dropped_duplicates.jsonl"
TOXIC_FILE = "output/dropped_toxic_and_ghosts.jsonl"

MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct" 
MAX_TOKENS = 4000 # Leaves a ~96 token safety buffer for the 4096 window

def hash_content(text):
    """Creates a normalized hash for deduplication."""
    clean_text = " ".join(text.lower().split())
    return hashlib.md5(clean_text.encode('utf-8')).hexdigest()

def sanitize_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ File not found: {INPUT_FILE}")
        return

    print(f"🚀 Initializing Sanitizer for: {INPUT_FILE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    seen_hashes = set()
    stats = {
        "processed": 0,
        "kept": 0,
        "duplicates": 0,
        "toxic": 0,
        "ghosts": 0
    }

    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(CLEAN_FILE, 'w', encoding='utf-8') as f_clean, \
         open(DUPLICATE_FILE, 'w', encoding='utf-8') as f_dup, \
         open(TOXIC_FILE, 'w', encoding='utf-8') as f_toxic:
        
        for line in f_in:
            if not line.strip(): continue
            stats["processed"] += 1
            
            try:
                data = json.loads(line)
                messages = data.get("messages", [])
            except json.JSONDecodeError:
                continue

            # --- 1. GHOST PROMPT CHECK ---
            user_content = next((m['content'] for m in messages if m['role'] == 'user'), "")
            if not user_content.strip():
                stats["ghosts"] += 1
                f_toxic.write(json.dumps({"reason": "GHOST_PROMPT", "data": data}) + "\n")
                continue

            # --- 2. TOXIC LENGTH CHECK ---
            try:
                full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                token_count = len(tokenizer.encode(full_text))
            except Exception:
                continue 
            
            if token_count > MAX_TOKENS:
                stats["toxic"] += 1
                f_toxic.write(json.dumps({"reason": f"TOXIC_LENGTH_{token_count}", "data": data}) + "\n")
                continue

            # --- 3. DEDUPLICATION CHECK ---
            assistant_text = next((m['content'] for m in messages if m['role'] == 'assistant'), "")
            if not assistant_text: continue 
            
            content_hash = hash_content(assistant_text)
            if content_hash in seen_hashes:
                stats["duplicates"] += 1
                f_dup.write(line)
                continue
            
            # --- 4. DATA PASSED ALL CHECKS ---
            seen_hashes.add(content_hash)
            f_clean.write(line)
            stats["kept"] += 1

    # --- FINAL REPORT ---
    print("\n" + "═"*50)
    print("🎯 FINAL SANITIZATION REPORT")
    print("═"*50)
    print(f"Total Processed     : {stats['processed']}")
    print(f"✅ Clean (Kept)      : {stats['kept']}")
    print(f"🗑️  Duplicates Cut   : {stats['duplicates']}")
    print(f"☢️  Toxic (>4000)    : {stats['toxic']}")
    print(f"👻 Ghost (Empty)    : {stats['ghosts']}")
    print("─" * 50)
    print(f"📂 Clean Dataset    -> {CLEAN_FILE}")
    print("═"*50)

if __name__ == "__main__":
    sanitize_dataset()