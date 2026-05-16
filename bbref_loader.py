"""
Basketball Reference game log loader using basketball_reference_web_scraper.

Provides:
- Per-game logs for current regular season (recent form)
- Per-game logs for playoffs (fills NBA Stats API gap)
- Always fetches fresh game log data — no game log caching
- BBRef player identifiers are cached permanently (they never change)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import re
from typing import Optional, Dict, List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

CACHE_DIR = Path("data/cache")
BBREF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
BBREF_ID_CACHE = CACHE_DIR / "bbref_identifiers.json"
BBREF_DELAY = 1.5  # seconds between BBRef requests


def _load_id_cache() -> Dict[str, str]:
    if BBREF_ID_CACHE.exists():
        try:
            with open(BBREF_ID_CACHE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_id_cache(cache: Dict[str, str]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(BBREF_ID_CACHE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def get_bbref_identifier(player_name: str) -> Optional[str]:
    """Return the Basketball Reference player identifier for a given name."""
    cache = _load_id_cache()
    if player_name in cache:
        return cache[player_name]

    try:
        from basketball_reference_web_scraper import client
        results = client.search(term=player_name)
        players = results.get("players", [])
        if not players:
            return None
        # Pick the best match — exact name match preferred
        match = next(
            (p for p in players if p["name"].lower() == player_name.lower()),
            players[0],
        )
        identifier = match["identifier"]
        cache[player_name] = identifier
        _save_id_cache(cache)
        time.sleep(BBREF_DELAY)
        return identifier
    except Exception:
        return None


def _parse_game_logs(raw: List[Dict]) -> pd.DataFrame:
    """Convert raw BBRef box score dicts to a clean DataFrame."""
    if not raw:
        return pd.DataFrame()

    rows = []
    for g in raw:
        if not g.get("active", True):
            continue
        secs = int(g.get("seconds_played", 0) or 0)
        if secs == 0:
            continue
        rows.append({
            "date":     str(g.get("date", "")),
            "opponent": str(g.get("opponent", "")).replace("Team.", ""),
            "location": str(g.get("location", "")).replace("Location.", ""),
            "outcome":  str(g.get("outcome", "")).replace("Outcome.", ""),
            "min":      round(secs / 60, 1),
            "pts":      int(g.get("points_scored", 0) or 0),
            "fgm":      int(g.get("made_field_goals", 0) or 0),
            "fga":      int(g.get("attempted_field_goals", 0) or 0),
            "fg3m":     int(g.get("made_three_point_field_goals", 0) or 0),
            "fg3a":     int(g.get("attempted_three_point_field_goals", 0) or 0),
            "ftm":      int(g.get("made_free_throws", 0) or 0),
            "fta":      int(g.get("attempted_free_throws", 0) or 0),
            "reb":      int(g.get("offensive_rebounds", 0) or 0) + int(g.get("defensive_rebounds", 0) or 0),
            "ast":      int(g.get("assists", 0) or 0),
            "stl":      int(g.get("steals", 0) or 0),
            "blk":      int(g.get("blocks", 0) or 0),
            "tov":      int(g.get("turnovers", 0) or 0),
            "plus_minus": int(g.get("plus_minus", 0) or 0),
        })
    return pd.DataFrame(rows)


def _log_cache_path(player_identifier: str, season_end_year: int, log_type: str) -> Path:
    return CACHE_DIR / f"gamelogs_{log_type}_{player_identifier}_{season_end_year}.json"


def _save_log_cache(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(df.to_dict(orient="records"), f, default=str)
    except Exception:
        pass


def _load_log_cache(path: Path) -> pd.DataFrame:
    try:
        with open(path) as f:
            data = json.load(f)
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def get_current_season_logs(
    player_name: str, season_end_year: int, cache_only: bool = False
) -> pd.DataFrame:
    """
    Fetch regular season game logs. Cached to disk permanently.
    cache_only=True: return from cache or empty — never hits BBRef.
    season_end_year: 2026 for the 2025-26 season.
    """
    identifier = get_bbref_identifier(player_name)
    if not identifier:
        return pd.DataFrame()
    cache_path = _log_cache_path(identifier, season_end_year, "regular")
    if cache_path.exists():
        return _load_log_cache(cache_path)
    if cache_only:
        return pd.DataFrame()
    try:
        from basketball_reference_web_scraper import client
        raw = client.regular_season_player_box_scores(
            player_identifier=identifier,
            season_end_year=season_end_year,
        )
        time.sleep(BBREF_DELAY)
        df = _parse_game_logs(raw)
        _save_log_cache(cache_path, df)
        return df
    except Exception:
        return pd.DataFrame()


def get_playoff_logs(
    player_name: str, season_end_year: int, cache_only: bool = False
) -> pd.DataFrame:
    """
    Fetch playoff game logs. Cached to disk permanently.
    cache_only=True: return from cache or empty — never hits BBRef.
    season_end_year: 2026 for the 2025-26 playoffs.
    """
    identifier = get_bbref_identifier(player_name)
    if not identifier:
        return pd.DataFrame()
    cache_path = _log_cache_path(identifier, season_end_year, "playoff")
    if cache_path.exists():
        return _load_log_cache(cache_path)
    if cache_only:
        return pd.DataFrame()
    try:
        from basketball_reference_web_scraper import client
        raw = client.playoff_player_box_scores(
            player_identifier=identifier,
            season_end_year=season_end_year,
        )
        time.sleep(BBREF_DELAY)
        df = _parse_game_logs(raw)
        _save_log_cache(cache_path, df)
        return df
    except Exception:
        return pd.DataFrame()


def prefetch_player_logs(player_names: List[str], season_end_year: int) -> int:
    """Pre-fetch and cache regular-season and playoff logs for a list of players.
    Skips players whose cache files already exist. Returns count newly fetched."""
    count = 0
    for name in player_names:
        identifier = get_bbref_identifier(name)
        if not identifier:
            continue
        for log_type, fetch_fn in [
            ("regular", lambda i: get_current_season_logs(name, season_end_year)),
            ("playoff", lambda i: get_playoff_logs(name, season_end_year)),
        ]:
            path = _log_cache_path(identifier, season_end_year, log_type)
            if not path.exists():
                fetch_fn(identifier)
                count += 1
    return count


def fmt_game_log_context(
    player_name: str,
    season_logs: pd.DataFrame,
    playoff_logs: pd.DataFrame,
) -> str:
    """Format game log data as LLM context string."""
    parts = []

    if not season_logs.empty:
        total = len(season_logs)

        def avg(col, df=season_logs): return df[col].mean()

        fg_pct = avg("fgm") / avg("fga") if avg("fga") > 0 else 0
        fg3_pct = avg("fg3m") / avg("fg3a") if avg("fg3a") > 0 else 0
        ft_pct = avg("ftm") / avg("fta") if avg("fta") > 0 else 0

        parts.append(
            f"{player_name} — full current regular season ({total} games, "
            f"source: Basketball Reference, refreshed on every ask):\n"
            f"  Season averages: PPG {avg('pts'):.1f} | FGA/g {avg('fga'):.1f} | "
            f"FG% {fg_pct:.1%} | 3PA/g {avg('fg3a'):.1f} | 3P% {fg3_pct:.1%} | "
            f"FTA/g {avg('fta'):.1f} | FT% {ft_pct:.1%} | "
            f"APG {avg('ast'):.1f} | RPG {avg('reb'):.1f} | "
            f"+/- {avg('plus_minus'):.1f}"
        )

        # Per-opponent FTA summary (surfaces hack-a patterns)
        if "opponent" in season_logs.columns:
            opp_fta = (
                season_logs.groupby("opponent")
                .agg(games=("fta", "count"), total_fta=("fta", "sum"), total_ftm=("ftm", "sum"))
                .assign(fta_pg=lambda d: d["total_fta"] / d["games"],
                        ft_pct=lambda d: d["total_ftm"] / d["total_fta"].clip(lower=1))
                .sort_values("fta_pg", ascending=False)
            )
            high_fta = opp_fta[opp_fta["fta_pg"] >= 2.0]
            if not high_fta.empty:
                lines = ["  FTA/g by opponent (teams where FTA/g ≥ 2.0 — potential hack targets):"]
                for opp, row in high_fta.iterrows():
                    opp_name = str(opp).replace("_", " ").title()
                    lines.append(
                        f"    vs {opp_name}: {row['fta_pg']:.1f} FTA/g "
                        f"({row['ft_pct']:.1%} FT%) over {int(row['games'])}g"
                    )
                parts.append("\n".join(lines))

        # Full per-game log
        parts.append(f"  Full game log:")
        for _, row in season_logs.iterrows():
            opp = row["opponent"].replace("_", " ").title()
            fga = row["fga"]
            fgm = row["fgm"]
            fg3a = row["fg3a"]
            fg3m = row["fg3m"]
            parts.append(
                f"    {row['date']} vs {opp} ({row['outcome']}): "
                f"{row['pts']}pts {fgm}/{fga}FG {fg3m}/{fg3a}3P "
                f"{row['ftm']}/{row['fta']}FT {row['ast']}ast {row['reb']}reb "
                f"+/-{row['plus_minus']}"
            )

    if not playoff_logs.empty:
        parts.append(
            f"\n{player_name} — playoffs ({len(playoff_logs)} games, "
            f"source: Basketball Reference):"
        )

        def pavg(col): return playoff_logs[col].mean()

        pfg_pct = pavg("fgm") / pavg("fga") if pavg("fga") > 0 else 0
        pfg3_pct = pavg("fg3m") / pavg("fg3a") if pavg("fg3a") > 0 else 0
        pft_pct = pavg("ftm") / pavg("fta") if pavg("fta") > 0 else 0

        parts.append(
            f"  Overall: PPG {pavg('pts'):.1f} | FGA/g {pavg('fga'):.1f} | "
            f"FG% {pfg_pct:.1%} | FTA/g {pavg('fta'):.1f} | FT% {pft_pct:.1%} | "
            f"3PA/g {pavg('fg3a'):.1f} | 3P% {pfg3_pct:.1%} | "
            f"APG {pavg('ast'):.1f} | +/- {pavg('plus_minus'):.1f}"
        )

        # Per-game playoff log
        parts.append(f"  Playoff game log:")
        for _, row in playoff_logs.iterrows():
            opp = row["opponent"].replace("_", " ").title()
            parts.append(
                f"    {row['date']} vs {opp} ({row['outcome']}): "
                f"{row['pts']}pts {row['fgm']}/{row['fga']}FG "
                f"{row['ftm']}/{row['fta']}FT {row['ast']}ast {row['reb']}reb "
                f"+/-{row['plus_minus']}"
            )

        # Series breakdown with FT% per series
        if "opponent" in playoff_logs.columns:
            parts.append(f"  Series averages:")
            for opp, grp in playoff_logs.groupby("opponent"):
                opp_name = opp.replace("_", " ").title()
                g = len(grp)
                s_ft_pct = grp["ftm"].sum() / grp["fta"].sum() if grp["fta"].sum() > 0 else 0
                parts.append(
                    f"    vs {opp_name} ({g}g): "
                    f"{grp['pts'].mean():.1f}pts "
                    f"{grp['fta'].mean():.1f}FTA/g ({s_ft_pct:.1%} FT%) "
                    f"{grp['ast'].mean():.1f}ast "
                    f"+/-{grp['plus_minus'].mean():.1f}"
                )

    return "\n".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Team stats & standings (Basketball Reference league page)
# ---------------------------------------------------------------------------

def _bbref_soup(url: str) -> Optional[BeautifulSoup]:
    """Fetch a BBRef page and return a BeautifulSoup with comments expanded."""
    try:
        r = requests.get(url, headers=BBREF_HEADERS, timeout=20)
        r.raise_for_status()
        r.encoding = "utf-8"
        # BBRef hides most tables in HTML comments — uncomment them
        html = re.sub(r"<!--(.*?)-->", r"\1", r.text, flags=re.DOTALL)
        return BeautifulSoup(html, "html.parser")
    except Exception:
        return None


_TEAM_STATS_TTL = 4 * 3600   # re-fetch at most every 4 hours
_BRACKET_TTL    = 30 * 60    # re-fetch at most every 30 minutes


def _team_stats_cache_path(season_end_year: int) -> Path:
    return CACHE_DIR / f"team_stats_{season_end_year}.json"


def _bracket_cache_path(season_end_year: int) -> Path:
    return CACHE_DIR / f"playoff_bracket_{season_end_year}.json"


def get_bbref_team_stats(season_end_year: int, force: bool = False) -> pd.DataFrame:
    """
    Scrape team advanced stats from Basketball Reference.
    Returns DataFrame with columns matching the existing app expectations:
    TEAM_NAME, OFF_RATING, DEF_RATING, NET_RATING, PACE, EFG_PCT,
    TM_TOV_PCT, OREB_PCT, TS_PCT, WINS, LOSSES, W, L, WinPct, Conference
    Cached to disk for up to 4 hours; pass force=True to bypass.
    """
    import time as _time
    cache_path = _team_stats_cache_path(season_end_year)
    if not force and cache_path.exists():
        try:
            if _time.time() - cache_path.stat().st_mtime < _TEAM_STATS_TTL:
                with open(cache_path) as f:
                    cached = json.load(f)
                if cached.get("season_end_year") == season_end_year and cached.get("rows"):
                    return pd.DataFrame(cached["rows"])
        except Exception:
            pass

    soup = _bbref_soup(f"https://www.basketball-reference.com/leagues/NBA_{season_end_year}.html")
    if soup is None:
        return pd.DataFrame()

    # Advanced team table has ORtg, DRtg, NRtg, Pace, eFG%, TOV%, ORB%, TS%
    table = soup.find("table", id="advanced-team")
    # Standings for conference + record
    east = soup.find("table", id="confs_standings_E")
    west = soup.find("table", id="confs_standings_W")

    if table is None:
        return pd.DataFrame()

    # Build conference lookup from standings
    conf_map: Dict[str, str] = {}
    seed_map: Dict[str, int] = {}
    for conf_label, tbl in [("East", east), ("West", west)]:
        if tbl is None:
            continue
        seed = 1
        for row in tbl.find("tbody").find_all("tr"):
            if row.get("class") and "thead" in row.get("class"):
                continue
            name_cell = row.find("td", {"data-stat": "team_name"})
            if name_cell:
                name = name_cell.get_text(strip=True).rstrip("*")
                conf_map[name] = conf_label
                seed_map[name] = seed
                seed += 1

    rows = []
    for row in table.find("tbody").find_all("tr"):
        if row.get("class"):
            continue
        def cell(stat):
            c = row.find(["td", "th"], {"data-stat": stat})
            return c.get_text(strip=True) if c else ""

        name = cell("team").rstrip("*").strip()
        if not name:
            continue

        def _f(s):
            try:
                return float(s.replace("+", ""))
            except (ValueError, AttributeError):
                return None

        rows.append({
            "TEAM_NAME":   name,
            "OFF_RATING":  _f(cell("off_rtg")),
            "DEF_RATING":  _f(cell("def_rtg")),
            "NET_RATING":  _f(cell("net_rtg")),
            "PACE":        _f(cell("pace")),
            "EFG_PCT":     _f(cell("efg_pct")),
            # BBRef stores tov_pct/orb_pct as whole numbers (13.2 = 13.2%)
            # but the display layer expects decimals (0.132) — divide by 100
            "TM_TOV_PCT":  (_f(cell("tov_pct")) / 100 if _f(cell("tov_pct")) is not None else None),
            "OREB_PCT":    (_f(cell("orb_pct")) / 100 if _f(cell("orb_pct")) is not None else None),
            "TS_PCT":      _f(cell("ts_pct")),
            "WINS":        _f(cell("wins")),
            "LOSSES":      _f(cell("losses")),
            "W":           _f(cell("wins")),
            "L":           _f(cell("losses")),
            "WinPct":      _f(cell("win_loss_pct")) if cell("win_loss_pct") else None,
            "Conference":  conf_map.get(name, ""),
            "PlayoffRank": seed_map.get(name, 99),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({"season_end_year": season_end_year, "rows": rows}, f)
        except Exception:
            pass
    return df


# ---------------------------------------------------------------------------
# Game box scores (Basketball Reference game pages)
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    # abbreviated forms
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# BBRef team abbreviations (a few differ from NBA API)
_BBREF_ABBR: Dict[str, str] = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}


def _parse_game_date(date_str: str, season_end_year: int) -> Optional[str]:
    """Convert 'April 19' or 'Apr 19' to 'YYYYMMDD' using season_end_year."""
    parts = date_str.strip().split()
    if len(parts) < 2:
        return None
    month_key = parts[0].lower().rstrip(".")
    month_num = _MONTH_MAP.get(month_key)
    if not month_num:
        return None
    try:
        day = int(parts[1])
    except ValueError:
        return None
    # Playoffs always fall in Apr-Jun of season_end_year
    year = season_end_year
    return f"{year}{month_num:02d}{day:02d}"


def _get_bbref_abbr(team_name: str) -> Optional[str]:
    """Return BBRef 3-letter abbreviation for a full team name."""
    abbr = _BBREF_ABBR.get(team_name)
    if abbr:
        return abbr
    # Fallback: match on last word (nickname)
    nick = team_name.split()[-1] if team_name else ""
    for name, a in _BBREF_ABBR.items():
        if name.endswith(nick):
            return a
    return None


def get_game_boxscore(
    date_str: str,
    home_team: str,
    away_team: str,
    season_end_year: int,
) -> Optional[Dict]:
    """
    Scrape a BBRef game box score page.
    Returns dict with keys: date, home, away, home_score, away_score,
    home_players, away_players (list of player stat dicts).
    Cached at data/cache/boxscore_{YYYYMMDD}_{HOMEABBR}.json
    """
    date_key = _parse_game_date(date_str, season_end_year)
    home_abbr = _get_bbref_abbr(home_team)
    away_abbr = _get_bbref_abbr(away_team)
    if not date_key or not home_abbr:
        return None

    cache_path = CACHE_DIR / f"boxscore_{date_key}_{home_abbr}.json"
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                data = json.load(f)
            # Derive scores from player totals when linescore was missing at scrape time
            if data.get("home_score") is None and data.get("home_players"):
                data["home_score"] = sum(p["pts"] for p in data["home_players"])
            if data.get("away_score") is None and data.get("away_players"):
                data["away_score"] = sum(p["pts"] for p in data["away_players"])
            return data
        except Exception:
            pass

    url = f"https://www.basketball-reference.com/boxscores/{date_key}0{home_abbr}.html"
    soup = _bbref_soup(url)
    if soup is None:
        return None

    result: Dict = {
        "date": date_str,
        "home": home_team,
        "away": away_team,
        "home_score": None,
        "away_score": None,
        "home_players": [],
        "away_players": [],
    }

    def _parse_box_table(abbr: str) -> List[Dict]:
        table = soup.find("table", id=f"box-{abbr}-game-basic")
        if table is None:
            return []
        players = []
        for row in table.find("tbody").find_all("tr"):
            if row.get("class") and any(c in row.get("class", []) for c in ["thead", "partial_table"]):
                continue
            name_cell = row.find("th", {"data-stat": "player"})
            if not name_cell:
                continue
            name = name_cell.get_text(strip=True)
            if not name or name.lower() in ("reserves", "team totals"):
                continue

            def _td(stat):
                c = row.find("td", {"data-stat": stat})
                return c.get_text(strip=True) if c else ""

            mp = _td("mp")
            # Skip DNP rows
            reason = _td("reason")
            if reason or not mp or mp == "Did Not Play" or mp == "Did Not Dress":
                continue
            try:
                mins = int(mp.split(":")[0]) if ":" in mp else int(mp)
            except (ValueError, AttributeError):
                mins = 0
            if mins < 1:
                continue

            def _int(s):
                try: return int(s)
                except (ValueError, TypeError): return 0

            players.append({
                "name": name,
                "min": mins,
                "pts": _int(_td("pts")),
                "reb": _int(_td("trb")),
                "ast": _int(_td("ast")),
                "fgm": _int(_td("fg")),
                "fga": _int(_td("fga")),
                "fg3m": _int(_td("fg3")),
                "fg3a": _int(_td("fg3a")),
                "ftm": _int(_td("ft")),
                "fta": _int(_td("fta")),
                "stl": _int(_td("stl")),
                "blk": _int(_td("blk")),
                "tov": _int(_td("tov")),
                "plus_minus": _int(_td("plus_minus")),
            })
        return players

    result["home_players"] = _parse_box_table(home_abbr)
    if away_abbr:
        result["away_players"] = _parse_box_table(away_abbr)

    # Extract final scores from the scoreline — match by nickname, abbreviation, or city
    linescore = soup.find("table", class_="linescore")
    if linescore:
        rows = linescore.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) >= 2:
                team_cell = cells[0].get_text(strip=True).lower()
                total_cell = cells[-1].get_text(strip=True)
                try:
                    score = int(total_cell)
                    away_tokens = [t.lower() for t in away_team.split()] + ([away_abbr.lower()] if away_abbr else [])
                    home_tokens = [t.lower() for t in home_team.split()] + ([home_abbr.lower()] if home_abbr else [])
                    if any(tok in team_cell for tok in away_tokens):
                        result["away_score"] = score
                    elif any(tok in team_cell for tok in home_tokens):
                        result["home_score"] = score
                except (ValueError, TypeError):
                    pass

    # Derive scores from player point totals when linescore parsing fails
    if result["home_score"] is None and result["home_players"]:
        result["home_score"] = sum(p["pts"] for p in result["home_players"])
    if result["away_score"] is None and result["away_players"]:
        result["away_score"] = sum(p["pts"] for p in result["away_players"])

    time.sleep(BBREF_DELAY)

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f)
    except Exception:
        pass

    return result


def fmt_boxscore(box: Dict) -> str:
    """Format a single game box score dict into LLM-readable text."""
    as_ = box.get("away_score")
    hs_ = box.get("home_score")
    away_score = f" {as_}" if as_ is not None else ""
    home_score = f" {hs_}" if hs_ is not None else ""
    if as_ is not None and hs_ is not None:
        winner = box["away"] if as_ > hs_ else box["home"]
        result_tag = f" — {winner.split()[-1].upper()} WIN"
    else:
        result_tag = ""
    lines = [
        f"{box['date']}: {box['away']}{away_score} @ {box['home']}{home_score}{result_tag}"
    ]

    for team_key, team_name in [("away_players", box["away"]), ("home_players", box["home"])]:
        players = box.get(team_key, [])
        if not players:
            continue
        lines.append(f"  {team_name}:")
        for p in sorted(players, key=lambda x: x["pts"], reverse=True):
            fg_str = f"{p['fgm']}/{p['fga']}FG"
            fg3_str = f" {p['fg3m']}/{p['fg3a']}3P" if p["fg3a"] > 0 else ""
            ft_str = f" {p['ftm']}/{p['fta']}FT" if p["fta"] > 0 else ""
            pm_val = p["plus_minus"]
            pm = f" ({'+' if pm_val > 0 else ''}{pm_val})" if pm_val != 0 else ""
            lines.append(
                f"    {p['name']}: {p['min']}min {p['pts']}pts {p['reb']}reb "
                f"{p['ast']}ast {fg_str}{fg3_str}{ft_str}{pm}"
            )
    return "\n".join(lines)


def _is_clean_game(g: Dict) -> bool:
    """Reject malformed bracket entries where the full schedule got concatenated into one row."""
    home = str(g.get("home", ""))
    away = str(g.get("away", ""))
    return "Game" not in home and "Game" not in away and len(home) <= 50 and len(away) <= 50


def get_playoff_series_boxscores(series: Dict, season_end_year: int) -> str:
    """
    Fetch and format full box scores for all played games in a playoff series.
    Returns formatted string for LLM context, empty string on failure.
    """
    games = [g for g in series.get("games", []) if g.get("played") and _is_clean_game(g)]
    if not games:
        return ""

    parts = [
        f"=== {series['team1'].upper()} vs {series['team2'].upper()} — "
        f"{series['round_name'].upper()} GAME-BY-GAME BOX SCORES ==="
    ]

    for i, g in enumerate(games, 1):
        box = get_game_boxscore(
            g["date"], g["home"], g["away"], season_end_year
        )
        if box:
            parts.append(f"\nGame {i} — {fmt_boxscore(box)}")
        else:
            # Fallback: just the score line we already have
            parts.append(
                f"\nGame {i} — {g['date']}: {g['away']} {g.get('away_score','')} "
                f"@ {g['home']} {g.get('home_score','')}"
            )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Playoff bracket (Basketball Reference playoffs page)
# ---------------------------------------------------------------------------

_ROUND_ORDER = {
    "First Round": 1,
    "Conference Semifinals": 2,
    "Conference Finals": 3,
    "Finals": 4,
    "NBA Finals": 4,
}


def get_bbref_playoff_bracket(season_end_year: int, force: bool = False) -> List[Dict]:
    """
    Scrape the live playoff bracket from Basketball Reference.
    Cached to disk for up to 30 minutes; pass force=True to bypass.
    Returns a list of series dicts:
      round_name, round_num, conf, team1, team2,
      wins1, wins2, status ('completed'|'in_progress'|'upcoming'),
      leader (team currently leading or winner),
      games (list of {date, away, home, away_score, home_score, played})
    Ordered most-recent round first.
    """
    import time as _time
    cache_path = _bracket_cache_path(season_end_year)
    if not force and cache_path.exists():
        try:
            if _time.time() - cache_path.stat().st_mtime < _BRACKET_TTL:
                with open(cache_path) as f:
                    cached = json.load(f)
                if cached.get("season_end_year") == season_end_year and cached.get("bracket"):
                    return cached["bracket"]
        except Exception:
            pass

    soup = _bbref_soup(
        f"https://www.basketball-reference.com/playoffs/NBA_{season_end_year}.html"
    )
    if soup is None:
        return []

    table = soup.find("table", id="all_playoffs")
    if table is None:
        return []

    series_list: List[Dict] = []
    current_series: Optional[Dict] = None

    for row in table.find_all("tr"):
        text = row.get_text(" ", strip=True)
        if not text:
            continue

        # Detect series header rows
        # Patterns: "Eastern Conference Semifinals New York Knicks lead Philadelphia 76ers (2-0)"
        #            "... over ... (4-3)"   "... tied with ... (1-1)"
        header_match = re.match(
            r"^(.*?(?:Conference (?:First Round|Semifinals|Finals)|(?:NBA )?Finals))\s+"
            r"(.+?)\s+(lead|over|tied with)\s+(.+?)\s+\((\d+)-(\d+)\)",
            text,
        )
        if header_match:
            round_conf = header_match.group(1).strip()
            team1      = header_match.group(2).strip()
            status_kw  = header_match.group(3).strip()
            team2      = header_match.group(4).strip()
            wins1      = int(header_match.group(5))
            wins2      = int(header_match.group(6))

            # Parse conference and round
            conf = "Finals"
            for label in ["Eastern", "Western"]:
                if label in round_conf:
                    conf = label
                    break
            round_name = round_conf.replace("Eastern Conference ", "").replace(
                "Western Conference ", ""
            ).strip()
            # Normalise
            if round_name == "NBA Finals" or round_name == "Finals":
                round_name = "Finals"
                conf = "Finals"

            status = (
                "completed"   if status_kw == "over" else
                "in_progress" if status_kw in ("lead", "tied with") else
                "upcoming"
            )
            leader = team1 if status == "completed" else (
                team1 if status_kw == "lead" else None
            )

            current_series = {
                "round_name": round_name,
                "round_num":  _ROUND_ORDER.get(round_name, 0),
                "conf":       conf,
                "team1":      team1,
                "team2":      team2,
                "wins1":      wins1,
                "wins2":      wins2,
                "status":     status,
                "leader":     leader,
                "games":      [],
            }
            series_list.append(current_series)
            continue

        # Detect upcoming series header (no score yet)
        upcoming_match = re.match(
            r"^(.*?(?:Conference (?:First Round|Semifinals|Finals)|(?:NBA )?Finals))\s+"
            r"(.+?)\s+vs\.\s+(.+)$",
            text,
        )
        if upcoming_match and current_series is None:
            round_conf = upcoming_match.group(1).strip()
            team1      = upcoming_match.group(2).strip()
            team2      = upcoming_match.group(3).strip()
            conf = "Finals"
            for label in ["Eastern", "Western"]:
                if label in round_conf:
                    conf = label
                    break
            round_name = round_conf.replace("Eastern Conference ", "").replace(
                "Western Conference ", ""
            ).strip()
            current_series = {
                "round_name": round_name,
                "round_num":  _ROUND_ORDER.get(round_name, 0),
                "conf":       conf,
                "team1":      team1,
                "team2":      team2,
                "wins1":      0,
                "wins2":      0,
                "status":     "upcoming",
                "leader":     None,
                "games":      [],
            }
            series_list.append(current_series)
            continue

        # Detect individual game rows
        # "Game 1 Mon, May 4 Philadelphia 76ers 98 @ New York Knicks 137"
        # "Game 3 Fri, May 8 New York Knicks @ Philadelphia 76ers"  (not yet played)
        if current_series is not None and re.match(r"^Game \d+", text):
            game_match = re.match(
                r"Game \d+\s+\w+,\s+(\w+ \d+)\s+(.+?)\s+(\d+)\s+@\s+(.+?)\s+(\d+)$",
                text,
            )
            if game_match:
                current_series["games"].append({
                    "date":       game_match.group(1),
                    "away":       game_match.group(2).strip(),
                    "home":       game_match.group(4).strip(),
                    "away_score": int(game_match.group(3)),
                    "home_score": int(game_match.group(5)),
                    "played":     True,
                })
            else:
                sched_match = re.match(
                    r"Game \d+\s+\w+,\s+(\w+ \d+)\s+(.+?)\s+@\s+(.+)$", text
                )
                if sched_match:
                    current_series["games"].append({
                        "date":       sched_match.group(1),
                        "away":       sched_match.group(2).strip(),
                        "home":       sched_match.group(3).strip(),
                        "away_score": None,
                        "home_score": None,
                        "played":     False,
                    })

    # Sort: in-progress first, then upcoming, then completed; within each by round desc
    order = {"in_progress": 0, "upcoming": 1, "completed": 2}
    series_list.sort(key=lambda s: (order[s["status"]], -s["round_num"]))
    if series_list:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({"season_end_year": season_end_year, "bracket": series_list}, f)
        except Exception:
            pass
    return series_list


def prefetch_playoff_boxscores(bracket: List[Dict], season_end_year: int) -> int:
    """
    Pre-fetch and cache box scores for every played game in the bracket.
    Skips games already in cache. Returns count of newly fetched games.
    """
    count = 0
    for series in bracket:
        for g in series.get("games", []):
            if not g.get("played") or not _is_clean_game(g):
                continue
            date_key = _parse_game_date(g["date"], season_end_year)
            home_abbr = _get_bbref_abbr(g["home"])
            if not date_key or not home_abbr:
                continue
            cache_path = CACHE_DIR / f"boxscore_{date_key}_{home_abbr}.json"
            if not cache_path.exists():
                get_game_boxscore(g["date"], g["home"], g["away"], season_end_year)
                count += 1
    return count


def get_cached_series_boxscores_for_teams(
    team_names: List[str],
    round_name: str = "Conference Semifinals",
) -> str:
    """
    Scan cached boxscore files for games between the specified teams.
    Used as fallback when the live bracket scraper hasn't yet indexed a new series.
    Returns a formatted string matching get_playoff_series_boxscores output.
    """
    if len(team_names) < 2:
        return ""

    matched_games: List[Dict] = []
    for cache_file in sorted(CACHE_DIR.glob("boxscore_*.json")):
        try:
            with open(cache_file) as f:
                box = json.load(f)
        except Exception:
            continue
        home = box.get("home", "")
        away = box.get("away", "")
        home_match = any(t.lower() in home.lower() for t in team_names)
        away_match = any(t.lower() in away.lower() for t in team_names)
        if not (home_match and away_match):
            continue
        # Derive scores from player totals if missing
        if box.get("home_score") is None and box.get("home_players"):
            box["home_score"] = sum(p["pts"] for p in box["home_players"])
        if box.get("away_score") is None and box.get("away_players"):
            box["away_score"] = sum(p["pts"] for p in box["away_players"])
        matched_games.append(box)

    if not matched_games:
        return ""

    # Identify the two teams
    all_teams = {g["home"] for g in matched_games} | {g["away"] for g in matched_games}
    team1, team2 = sorted(all_teams)[:2]

    # Count series wins
    wins: Dict[str, int] = {t: 0 for t in all_teams}
    for g in matched_games:
        hs = g.get("home_score") or 0
        as_ = g.get("away_score") or 0
        winner = g["home"] if hs > as_ else g["away"]
        wins[winner] = wins.get(winner, 0) + 1

    w1, w2 = wins.get(team1, 0), wins.get(team2, 0)
    if w1 > w2:
        series_status = f"{team1} lead {w1}-{w2}"
    elif w2 > w1:
        series_status = f"{team2} lead {w2}-{w1}"
    else:
        series_status = f"Tied {w1}-{w2}"

    parts = [
        f"=== {team1.upper()} vs {team2.upper()} — {round_name.upper()} BOX SCORES ===",
        f"Series: {series_status}",
    ]
    for i, box in enumerate(matched_games, 1):
        parts.append(f"\nGame {i} — {fmt_boxscore(box)}")

    return "\n".join(parts)


def fmt_playoff_context(
    bracket: List[Dict],
    filter_teams: Optional[List[str]] = None,
) -> str:
    """
    Format playoff bracket data as LLM context.
    filter_teams: if provided, only include series involving those teams.
    Returns empty string if bracket is empty.
    """
    if not bracket:
        return ""

    # Group by round for cleaner output
    from collections import defaultdict
    by_round: Dict[str, List[Dict]] = defaultdict(list)
    for s in bracket:
        key = f"{s['conf']} — {s['round_name']}"
        by_round[key].append(s)

    # If filtering, drop series with no matching team
    def _matches(s: Dict) -> bool:
        if not filter_teams:
            return True
        t1, t2 = s["team1"].lower(), s["team2"].lower()
        return any(ft.lower() in t1 or ft.lower() in t2 for ft in filter_teams)

    lines = ["=== 2025-26 NBA PLAYOFFS ==="]
    for section, series_in_round in by_round.items():
        matching = [s for s in series_in_round if _matches(s)]
        if not matching:
            continue
        lines.append(f"\n{section.upper()}")
        for s in matching:
            w1, w2 = s["wins1"], s["wins2"]
            status = s["status"]
            if status == "completed":
                winner = s["leader"] or (s["team1"] if w1 > w2 else s["team2"])
                result = f"{winner} win {max(w1,w2)}-{min(w1,w2)}"
            elif status == "in_progress":
                leader = s.get("leader")
                result = f"{leader} lead {w1}-{w2}" if leader else f"Tied {w1}-{w2}"
            else:
                result = "Upcoming"
            lines.append(f"  {s['team1']} vs {s['team2']}: {result}")
            for g in s.get("games", []):
                if not _is_clean_game(g):
                    continue
                if g.get("played"):
                    as_ = g.get("away_score")
                    hs_ = g.get("home_score")
                    if as_ is not None and hs_ is not None:
                        winner = g["away"] if as_ > hs_ else g["home"]
                        win_tag = f" ({winner.split()[-1].upper()} WIN)"
                    else:
                        win_tag = ""
                    lines.append(
                        f"    {g['date']}: {g['away']} {as_} @ "
                        f"{g['home']} {hs_}{win_tag}"
                    )
                else:
                    lines.append(f"    {g['date']}: {g['away']} @ {g['home']} (scheduled)")

    return "\n".join(lines) if len(lines) > 1 else ""
