"""
merge_dataset.py
----------------
Merges human captions + AI captions into one clean dataset.

Input files:
    data/raw/manual_captions.csv   — columns: text, source, label
    data/raw/ai_captions.csv       — columns: text, source, label

Output:
    data/processed/dataset.csv     — final clean dataset ready for model

Usage:
    python src/merge_dataset.py
"""

import pandas as pd
import re
import os
from datetime import datetime

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
HUMAN_PATH  = "data/raw/manual_captions.csv"
AI_PATH     = "data/raw/ai_captions.csv"
OUTPUT_PATH = "data/processed/dataset.csv"

# ─────────────────────────────────────────────
# INDICATOR FUNCTIONS
# ─────────────────────────────────────────────

BUZZWORDS = [
    "utilize", "delve", "comprehensive", "nuanced", "tapestry",
    "elevate", "embark", "foster", "intricate", "testament",
    "vibrant", "beacon", "pivotal", "seamless", "leverage",
    "boundaries", "realm", "unleash", "curated", "innovative"
]

STARTER_VERBS = [
    "embrace", "discover", "transform", "explore", "celebrate",
    "unleash", "elevate", "unlock", "dive", "ignite", "inspire",
    "capture", "find", "let", "join", "experience"
]

EMOJI_PATTERN = re.compile(
    "[\U00010000-\U0010ffff"
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "]+", flags=re.UNICODE
)

def compute_all_indicators(df):
    df["word_count"]       = df["text"].apply(lambda t: len(str(t).split()))
    df["char_count"]       = df["text"].apply(lambda t: len(str(t)))
    df["has_emoji"]        = df["text"].apply(lambda t: "yes" if EMOJI_PATTERN.search(str(t)) else "no")
    df["has_em_dash"]      = df["text"].apply(lambda t: "yes" if ("—" in str(t) or "–" in str(t)) else "no")
    df["has_ellipsis"]     = df["text"].apply(lambda t: "yes" if ("..." in str(t) or "…" in str(t)) else "no")
    df["starts_with_verb"] = df["text"].apply(
        lambda t: "yes" if (str(t).strip().split()[0].lower().rstrip(".,!?") in STARTER_VERBS
                           if str(t).strip() else False) else "no"
    )
    df["has_buzzwords"]    = df["text"].apply(
        lambda t: "yes" if any(w in str(t).lower() for w in BUZZWORDS) else "no"
    )
    return df

def clean_text(text):
    text = str(text).replace("\n", " ").replace("\r", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def standardize_source(source):
    """Normalize source names to consistent format."""
    s = str(source).strip().lower()
    if "chatgpt" in s or "chat gpt" in s or "gpt" in s:
        return "ai_chatgpt"
    elif "gemini" in s:
        return "ai_gemini"
    elif "copilot" in s:
        return "ai_copilot"
    elif "perplexity" in s:
        return "ai_perplexity"
    elif "instagram" in s or "manual" in s or "human" in s:
        return "human_instagram"
    else:
        return source

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def merge():
    print("\n─── AuthentiScan Dataset Merger ───\n")
    os.makedirs("data/processed", exist_ok=True)

    # ── Load Human Captions ──
    print("Loading human captions...")
    human_df = pd.read_csv(HUMAN_PATH, encoding="utf-8-sig", low_memory=False)
    human_df.columns = human_df.columns.str.strip().str.lower()
    human_df = human_df.rename(columns={"captions": "text"})
    human_df["label"]  = 0
    human_df["source"] = human_df["source"].apply(standardize_source)
    print(f"  ✓ {len(human_df)} human captions loaded")

    # ── Load AI Captions ──
    print("Loading AI captions...")
    ai_df = pd.read_csv(AI_PATH, encoding="utf-8-sig", low_memory=False)
    ai_df.columns = ai_df.columns.str.strip().str.lower()
    ai_df = ai_df.rename(columns={"captions": "text"})
    ai_df["label"]  = 1
    ai_df["source"] = ai_df["source"].apply(standardize_source)
    print(f"  ✓ {len(ai_df)} AI captions loaded")
    print(f"     breakdown: {ai_df['source'].value_counts().to_dict()}")

    # ── Merge ──
    merged = pd.concat([human_df, ai_df], ignore_index=True)

    # ── Clean Text ──
    merged["text"] = merged["text"].apply(clean_text)

    # ── Drop empty rows ──
    merged = merged[merged["text"].str.strip() != ""]
    merged = merged.dropna(subset=["text"])

    # ── Compute all indicator columns ──
    print("\nComputing indicator columns...")
    merged = compute_all_indicators(merged)

    # ── Filter word count ──
    before = len(merged)
    merged = merged[merged["word_count"] >= 5]
    after  = len(merged)
    if before != after:
        print(f"  → Removed {before - after} captions under 5 words")

    # ── Drop duplicates ──
    before = len(merged)
    merged = merged.drop_duplicates(subset=["text"])
    after  = len(merged)
    if before != after:
        print(f"  → Removed {before - after} duplicate captions")

    # ── Balance to 50-50 ──
    human_clean = merged[merged['label'] == 0]
    ai_clean    = merged[merged['label'] == 1]
    min_count   = min(len(human_clean), len(ai_clean))
    human_balanced = human_clean.sample(min_count, random_state=42)
    ai_balanced    = ai_clean.sample(min_count, random_state=42)
    merged = pd.concat([human_balanced, ai_balanced], ignore_index=True)
    print(f"  → Balanced to {min_count} human + {min_count} AI = {min_count*2} total")

    # ── Shuffle randomly ──
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  → Dataset shuffled randomly")

    # ── Reset IDs ──
    merged["id"] = range(1, len(merged) + 1)
    merged["collection_date"] = datetime.today().strftime("%Y-%m-%d")

    # ── Final column order ──
    final_columns = [
        "id", "text", "label", "source",
        "word_count", "char_count",
        "has_emoji", "has_em_dash", "has_ellipsis",
        "starts_with_verb", "has_buzzwords",
        "collection_date"
    ]
    merged = merged[final_columns]

    # ── Save ──
    merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    # ── Summary ──
    print(f"\n✅ Dataset saved to {OUTPUT_PATH}")
    print(f"\n{'─'*40}")
    print(f"  Total captions     : {len(merged)}")
    print(f"  Human (label=0)    : {len(merged[merged['label']==0])}")
    print(f"  AI    (label=1)    : {len(merged[merged['label']==1])}")
    print(f"\n  Source breakdown:")
    print(merged['source'].value_counts().to_string())
    print(f"\n  Word count stats:")
    print(merged['word_count'].describe().round(1).to_string())
    print(f"\n  Has emoji          : {(merged['has_emoji']=='yes').sum()}")
    print(f"  Has em dash        : {(merged['has_em_dash']=='yes').sum()}")
    print(f"  Has ellipsis       : {(merged['has_ellipsis']=='yes').sum()}")
    print(f"  Has buzzwords      : {(merged['has_buzzwords']=='yes').sum()}")
    print(f"{'─'*40}")

if __name__ == "__main__":
    merge()