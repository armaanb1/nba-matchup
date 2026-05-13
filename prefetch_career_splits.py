"""
Pre-fetch career splits for all active NBA players and save to cache.
Run locally before deploying: python prefetch_career_splits.py
Then commit data/cache/ to git so Streamlit Cloud has everything pre-loaded.
"""
import time
from pathlib import Path
from nba_api.stats.static import players as nba_players_static
from data_loader import get_player_career_splits, CACHE_DIR

def main():
    active = nba_players_static.get_active_players()
    total = len(active)
    skipped = 0
    fetched = 0
    failed = 0

    print(f"Pre-fetching career splits for {total} active players...")
    print(f"Cache dir: {CACHE_DIR}\n")

    for i, player in enumerate(active, 1):
        pid = player["id"]
        name = player["full_name"]
        proc_cache = CACHE_DIR / f"career_splits_processed_{pid}.json"

        # Skip if processed cache already exists
        if proc_cache.exists():
            skipped += 1
            print(f"[{i}/{total}] SKIP  {name}")
            continue

        print(f"[{i}/{total}] Fetching {name}...", end=" ", flush=True)
        try:
            df, _ = get_player_career_splits(pid)
            if df is not None and not df.empty:
                fetched += 1
                print(f"OK ({len(df)} seasons)")
            else:
                failed += 1
                print("EMPTY")
        except Exception as e:
            failed += 1
            print(f"FAILED: {e}")

        # Respect NBA API rate limiting
        time.sleep(1.2)

    print(f"\nDone. Fetched: {fetched}  Skipped (cached): {skipped}  Failed: {failed}")
    print(f"\nNow run:")
    print(f"  git add data/cache/career_splits_*.json")
    print(f"  git commit -m 'Pre-fetch career splits for all active NBA players'")
    print(f"  git push")

if __name__ == "__main__":
    main()
