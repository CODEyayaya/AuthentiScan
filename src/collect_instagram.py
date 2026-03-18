"""
collect_instagram.py
--------------------
Scrapes captions from public Instagram accounts using Instaloader.
Saves results to data/raw/instaloader_captions.csv

Usage:
    python src/collect_instagram.py

Requirements:
    pip install instaloader
"""

import instaloader
import csv
import os
import re
from datetime import datetime

# ─────────────────────────────────────────────
#  EDIT THIS LIST — add any public IG usernames
# ─────────────────────────────────────────────
ACCOUNTS = [
    # Travel
    "travelandleisure",
    "traveltomtom",
    # Food
    "blackforager",
    "davidchang",
    # Fitness
    "thebodycoach",
    "eddiehallwsm",
    # Motivation
    "garyvee",
    "priyankachopra",
    # Nature
    "tiffpenguin",
    "fosterhunting",
    # Fashion / Lifestyle
    "jihoon",
    "rida.tharanaa"
]

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MAX_POSTS_PER_ACCOUNT = 60       # how many posts to pull per account
MIN_WORDS = 30                   # filter: minimum word count
MAX_WORDS = 150                  # filter: maximum word count
OUTPUT_PATH = "data/raw/instaloader_captions.csv"

# AI buzzwords to flag
BUZZWORDS = [
    "utilize", "delve", "comprehensive", "nuanced", "tapestry",
    "elevate", "embark", "foster", "intricate", "testament",
    "vibrant", "beacon", "pivotal", "seamless", "leverage"
]

# Verbs commonly used at the start of AI captions
STARTER_VERBS = [
    "embrace", "discover", "transform", "explore", "celebrate",
    "unleash", "elevate", "unlock", "dive", "ignite", "inspire"
]


# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────

def count_words(text):
    return len(text.split())

def has_emoji(text):
    emoji_pattern = re.compile(
        "[\U00010000-\U0010ffff"
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE
    )
    return "yes" if emoji_pattern.search(text) else "no"

def has_em_dash(text):
    return "yes" if "—" in text or "–" in text else "no"

def has_ellipsis(text):
    return "yes" if "..." in text or "…" in text else "no"

def starts_with_verb(text):
    first_word = text.strip().split()[0].lower().rstrip(".,!?") if text.strip() else ""
    return "yes" if first_word in STARTER_VERBS else "no"

def has_buzzword(text):
    text_lower = text.lower()
    return "yes" if any(word in text_lower for word in BUZZWORDS) else "no"

def clean_text(text):
    # Remove newlines, extra whitespace — keep emojis and punctuation
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def guess_topic(text, username):
    """Rough topic guess based on account or keywords in caption."""
    text_lower = text.lower()
    username_lower = username.lower()

    if any(w in username_lower for w in ["travel", "destination", "natgeo"]):
        return "travel"
    elif any(w in username_lower for w in ["food", "baker", "cook", "eat"]):
        return "food"
    elif any(w in username_lower for w in ["fit", "gym", "sport", "nike", "rock"]):
        return "fitness"
    elif any(w in username_lower for w in ["zen", "goal", "daily", "motivat"]):
        return "motivation"
    elif any(w in text_lower for w in ["nature", "forest", "mountain", "ocean", "sky"]):
        return "nature"
    elif any(w in text_lower for w in ["food", "recipe", "cook", "eat", "delicious"]):
        return "food"
    elif any(w in text_lower for w in ["travel", "trip", "journey", "explore", "adventure"]):
        return "travel"
    else:
        return "general"


# ─────────────────────────────────────────────
#  MAIN SCRAPER
# ─────────────────────────────────────────────

def scrape_accounts(accounts, max_posts, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    loader = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        quiet=True
    )

    rows = []
    id_counter = 1

    for username in accounts:
        print(f"\n→ Scraping @{username} ...")
        try:
            profile = instaloader.Profile.from_username(loader.context, username)
            post_count = 0

            for post in profile.get_posts():
                if post_count >= max_posts:
                    break

                caption = post.caption
                if not caption:
                    continue

                caption = clean_text(caption)
                wc = count_words(caption)

                # Word count filter
                if wc < MIN_WORDS or wc > MAX_WORDS:
                    continue

                # Basic English check — skip if too many non-ASCII chars
                ascii_ratio = sum(1 for c in caption if ord(c) < 128) / len(caption)
                if ascii_ratio < 0.7:
                    continue

                rows.append({
                    "id": id_counter,
                    "text": caption,
                    "label": 0,
                    "source": "human_instagram",
                    "word_count": wc,
                    "char_count": len(caption),
                    "topic": guess_topic(caption, username),
                    "has_emoji": has_emoji(caption),
                    "has_em_dash": has_em_dash(caption),
                    "has_ellipsis": has_ellipsis(caption),
                    "starts_with_verb": starts_with_verb(caption),
                    "has_buzzwords": has_buzzword(caption),
                    "language": "en",
                    "flagged": "no",
                    "model_version": "",
                    "prompt_used": "",
                    "collection_date": datetime.today().strftime("%Y-%m-%d"),
                    "instagram_account": username
                })

                id_counter += 1
                post_count += 1

            print(f"   ✓ Collected {post_count} captions from @{username}")

        except instaloader.exceptions.ProfileNotExistsException:
            print(f"   ✗ @{username} not found — skipping")
        except Exception as e:
            print(f"   ✗ Error on @{username}: {e} — skipping")

    # Write CSV
    if not rows:
        print("\n⚠ No captions collected. Check your account list or internet connection.")
        return

    fieldnames = [
        "id", "text", "label", "source", "word_count", "char_count",
        "topic", "has_emoji", "has_em_dash", "has_ellipsis",
        "starts_with_verb", "has_buzzwords", "language", "flagged",
        "model_version", "prompt_used", "collection_date", "instagram_account"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Done! {len(rows)} captions saved to {output_path}")


# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    scrape_accounts(ACCOUNTS, MAX_POSTS_PER_ACCOUNT, OUTPUT_PATH)