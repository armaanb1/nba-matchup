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


def get_current_season_logs(player_name: str, season_end_year: int) -> pd.DataFrame:
    """
    Fetch regular season game logs. Always fresh — no caching.
    season_end_year: 2026 for the 2025-26 season.
    """
    identifier = get_bbref_identifier(player_name)
    if not identifier:
        return pd.DataFrame()
    try:
        from basketball_reference_web_scraper import client
        raw = client.regular_season_player_box_scores(
            player_identifier=identifier,
            season_end_year=season_end_year,
        )
        time.sleep(BBREF_DELAY)
        return _parse_game_logs(raw)
    except Exception:
        return pd.DataFrame()


def get_playoff_logs(player_name: str, season_end_year: int) -> pd.DataFrame:
    """
    Fetch playoff game logs. Always fresh — no caching.
    season_end_year: 2026 for the 2025-26 playoffs.
    """
    identifier = get_bbref_identifier(player_name)
    if not identifier:
        return pd.DataFrame()
    try:
        from basketball_reference_web_scraper import client
        raw = client.playoff_player_box_scores(
            player_identifier=identifier,
            season_end_year=season_end_year,
        )
        time.sleep(BBREF_DELAY)
        return _parse_game_logs(raw)
    except Exception:
        return pd.DataFrame()


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
        # BBRef hides most tables in HTML comments — uncomment them
        html = re.sub(r"<!--(.*?)-->", r"\1", r.text, flags=re.DOTALL)
        return BeautifulSoup(html, "html.parser")
    except Exception:
        return None


def get_bbref_team_stats(season_end_year: int) -> pd.DataFrame:
    """
    Scrape team advanced stats from Basketball Reference.
    Returns DataFrame with columns matching the existing app expectations:
    TEAM_NAME, OFF_RATING, DEF_RATING, NET_RATING, PACE, EFG_PCT,
    TM_TOV_PCT, OREB_PCT, TS_PCT, WINS, LOSSES, W, L, WinPct, Conference
    Always fresh — no caching.
    """
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

    return pd.DataFrame(rows)


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


def get_bbref_playoff_bracket(season_end_year: int) -> List[Dict]:
    """
    Scrape the live playoff bracket from Basketball Reference.
    Returns a list of series dicts, always fresh:
      round_name, round_num, conf, team1, team2,
      wins1, wins2, status ('completed'|'in_progress'|'upcoming'),
      leader (team currently leading or winner),
      games (list of {date, away, home, away_score, home_score, played})
    Ordered most-recent round first.
    """
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
    return series_list
