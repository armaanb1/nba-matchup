"""
Data loading, caching, and player enrichment.

Sources:
  1. NBA Stats API (nba_api)  — matchup data, player bio, season stats
  2. Basketball Reference      — advanced stats (PER, BPM, VORP, Win Shares)

All API responses are cached to data/cache/ to avoid repeated slow fetches.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
from nba_api.stats.endpoints import (
    commonplayerinfo,
    leaguedashplayerstats,
    leaguedashteamstats,
    leagueseasonmatchups,
    leaguestandings,
    playercareerstats,
)
from nba_api.stats.static import players as nba_players_static

from models import MatchupGraph, Player

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NBA_API_DELAY = 1.0   # seconds between NBA API calls

# ---------------------------------------------------------------------------
# Direct NBA.com Stats API — headers required to avoid 403
# ---------------------------------------------------------------------------

# Full header set matching what nba_api uses internally — required to avoid 403/timeouts
_NBA_DIRECT_HEADERS = {
    "Host": "stats.nba.com",
    "Connection": "keep-alive",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

# Shared session for connection pooling
_nba_session: Optional[requests.Session] = None


def _get_nba_session() -> requests.Session:
    global _nba_session
    if _nba_session is None:
        _nba_session = requests.Session()
        _nba_session.headers.update(_NBA_DIRECT_HEADERS)
        _nba_session.max_redirects = 5   # treat redirect loop as failure, not infinite loop
    return _nba_session


def _fetch_nba_direct(
    url: str,
    cache_path: Path,
    force_refresh: bool = False,
    timeout: int = 60,
) -> Optional[Dict]:
    """
    Fetch a raw NBA.com stats API URL, with file-based caching.
    Retries once after 3 seconds on failure.
    Falls back to stale cache if both attempts fail.
    Returns parsed JSON or None on total failure.
    """
    if not force_refresh and cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            pass

    global _nba_session
    session = _get_nba_session()
    for attempt in range(2):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            with open(cache_path, "w") as f:
                json.dump(data, f)
            return data
        except requests.exceptions.TooManyRedirects:
            # Redirect loop = endpoint not available yet (e.g. playoffs not started)
            print(f"  Redirect loop for {url} — endpoint not available yet, returning None")
            return None
        except Exception as e:
            print(f"  Direct fetch attempt {attempt + 1} failed ({url}): {e}")
            if attempt == 0:
                time.sleep(3)
                # Re-create session on retry in case of connection issue
                _nba_session = None

    # Return stale cached data as fallback (even if force_refresh was True)
    if cache_path.exists():
        try:
            print(f"  Falling back to stale cache for {cache_path.name}")
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _parse_nba_result_set(data: Dict, idx: int = 0) -> pd.DataFrame:
    """Parse resultSets[idx] from a raw NBA.com JSON response into a DataFrame.
    Column names are read dynamically from the headers key — never assumed by position.
    """
    try:
        rs = data["resultSets"][idx]
        return pd.DataFrame(rs["rowSet"], columns=rs["headers"])
    except Exception as e:
        print(f"  Result set parse failed (idx={idx}): {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Player lookup
# ---------------------------------------------------------------------------

def find_nba_player(name: str) -> Optional[Dict]:
    """Find player in the nba_api static list (case-insensitive partial match)."""
    all_players = nba_players_static.get_players()
    nl = name.strip().lower()
    exact = [p for p in all_players if p["full_name"].lower() == nl]
    if exact:
        return exact[0]
    partial = [p for p in all_players if nl in p["full_name"].lower()]
    return partial[0] if partial else None


# ---------------------------------------------------------------------------
# Matchup data
# ---------------------------------------------------------------------------

def load_matchup_data(season: str = "2024-25", season_type: str = "Regular Season",
                      min_possessions: int = 20) -> pd.DataFrame:
    """
    Load matchup data, using a CSV cache if available.
    Returns a filtered DataFrame with at least min_possessions per edge.
    """
    cache_path = CACHE_DIR / f"matchups_{season.replace('-','_')}_{season_type.replace(' ','_')}.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        df = _fetch_matchup_data(season, season_type)
        df.to_csv(cache_path, index=False)

    return df[df["PARTIAL_POSS"] >= min_possessions].copy()


def _fetch_matchup_data(season: str, season_type: str) -> pd.DataFrame:
    matchups = leagueseasonmatchups.LeagueSeasonMatchups(
        league_id="00",
        per_mode_simple="Totals",
        season=season,
        season_type_playoffs=season_type,
    )
    time.sleep(NBA_API_DELAY)
    return matchups.get_data_frames()[0]


# ---------------------------------------------------------------------------
# Player bio (CommonPlayerInfo)
# ---------------------------------------------------------------------------

def get_player_bio(player_id: int) -> Dict:
    cache_path = CACHE_DIR / f"bio_{player_id}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        time.sleep(NBA_API_DELAY)
        df = info.get_data_frames()[0]
        if df.empty:
            return {}

        row = df.iloc[0]
        bio = {
            "position": str(row.get("POSITION", "") or ""),
            "team": str(row.get("TEAM_NAME", "") or ""),
            "team_abbreviation": str(row.get("TEAM_ABBREVIATION", "") or ""),
            "height": str(row.get("HEIGHT", "") or ""),
            "weight": _safe_int(row.get("WEIGHT")),
            "birthdate": str(row.get("BIRTHDATE", "") or "")[:10],
            "experience": _safe_int(row.get("SEASON_EXP")),
            "jersey": str(row.get("JERSEY", "") or ""),
            "draft_year": str(row.get("DRAFT_YEAR", "") or ""),
            "draft_round": str(row.get("DRAFT_ROUND", "") or ""),
            "draft_pick": str(row.get("DRAFT_NUMBER", "") or ""),
        }

        with open(cache_path, "w") as f:
            json.dump(bio, f)
        return bio

    except Exception as e:
        print(f"  Bio fetch failed for player {player_id}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Per-game season stats (PlayerCareerStats + LeagueDashPlayerStats)
# ---------------------------------------------------------------------------

def get_player_season_stats(player_id: int, season: str = "2024-25") -> Dict:
    cache_path = CACHE_DIR / f"stats_{player_id}_{season.replace('-','_')}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    stats: Dict = {}
    try:
        career = playercareerstats.PlayerCareerStats(player_id=player_id,
                                                     per_mode36="PerGame")
        time.sleep(NBA_API_DELAY)
        df = career.get_data_frames()[0]
        row_df = df[df["SEASON_ID"] == season]
        if not row_df.empty:
            row = row_df.iloc[-1]  # take most recent team row if traded
            stats.update({
                "ppg": _safe_float(row.get("PTS")),
                "rpg": _safe_float(row.get("REB")),
                "apg": _safe_float(row.get("AST")),
                "spg": _safe_float(row.get("STL")),
                "bpg": _safe_float(row.get("BLK")),
                "tov": _safe_float(row.get("TOV")),
                "mpg": _safe_float(row.get("MIN")),
                "fg_pct": _safe_float(row.get("FG_PCT")),
                "fg3_pct": _safe_float(row.get("FG3_PCT")),
                "ft_pct": _safe_float(row.get("FT_PCT")),
                "games": _safe_int(row.get("GP")),
            })
    except Exception as e:
        print(f"  Career stats failed for player {player_id}: {e}")

    with open(cache_path, "w") as f:
        json.dump(stats, f)
    return stats


# ---------------------------------------------------------------------------
# Basketball Reference — Advanced Stats
# ---------------------------------------------------------------------------
# NBA API Advanced Stats
# ---------------------------------------------------------------------------

def _fetch_all_player_advanced_stats(season: str) -> pd.DataFrame:
    """
    Fetch advanced stats for ALL players in one API call and cache as a single file.
    Returns the full DataFrame keyed by PLAYER_ID.
    """
    cache_path = CACHE_DIR / f"adv_all_{season.replace('-','_')}.json"
    if cache_path.exists():
        try:
            return pd.read_json(cache_path)
        except Exception:
            pass

    try:
        dash = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
        )
        time.sleep(NBA_API_DELAY)
        df = dash.get_data_frames()[0]
        if not df.empty:
            df.to_json(cache_path, orient="records")
        return df
    except Exception as e:
        print(f"  Bulk advanced stats fetch failed: {e}")
        return pd.DataFrame()


def get_nba_advanced_stats(player_id: int, season: str = "2024-25") -> Dict:
    """
    Look up a single player's advanced stats from the bulk-fetched DataFrame.
    Returns off/def/net rating, USG%, PIE, AST%.
    """
    all_df = _fetch_all_player_advanced_stats(season)
    if all_df.empty:
        return {}

    row_df = all_df[all_df["PLAYER_ID"] == player_id]
    if row_df.empty:
        return {}

    row = row_df.iloc[0]
    return {
        "off_rating": _safe_float(row.get("OFF_RATING")),
        "def_rating": _safe_float(row.get("DEF_RATING")),
        "net_rating": _safe_float(row.get("NET_RATING")),
        "usg_pct": _safe_float(row.get("USG_PCT")),
        "pie": _safe_float(row.get("PIE")),
        "ast_pct": _safe_float(row.get("AST_PCT")),
        "ts_pct": _safe_float(row.get("TS_PCT")),
    }


# ---------------------------------------------------------------------------
# EPM data from dunksandthrees.com
# ---------------------------------------------------------------------------

def fetch_epm_data(season: int = 2026, force_refresh: bool = False) -> Dict[str, dict]:
    """
    Fetch EPM and per-100 stats from dunksandthrees.com/epm.
    Data is embedded as JS object literals in the SSR HTML.
    Returns dict keyed by lowercase player name -> stat dict.
    Cached to data/cache/epm_{season}.json.
    """
    import re
    cache_path = CACHE_DIR / f"epm_{season}.json"
    if not force_refresh and cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            pass

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = requests.get(
            "https://dunksandthrees.com/epm",
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"  EPM fetch failed: {e}")
        return {}

    html = resp.text

    float_fields = [
        "off", "def", "tot",
        "p_pts_100", "p_ast_100", "p_blk_100", "p_stl_100",
        "p_drb_100", "p_orb_100", "p_tov_100",
        "p_fga_rim_100", "p_fga_mid_100", "p_fg3a_100",
        "p_fgpct_rim", "p_fgpct_mid",
    ]

    result = {}

    # Strategy 1: find player_name occurrences and extract surrounding context
    # Works for both JS literal and JSON formats
    for name_match in re.finditer(r'player_name["\s]*[:=]["\s]*"([^"]+)"', html):
        player_name = name_match.group(1).strip()
        # Take a window of text around the name to find sibling fields
        start = max(0, name_match.start() - 500)
        end = min(len(html), name_match.end() + 1500)
        window = html[start:end]

        stats = {}
        for field in float_fields:
            m = re.search(
                rf'[,\{{]?\s*"?{re.escape(field)}"?\s*:\s*(-?[\d]+(?:\.[\d]+)?)',
                window
            )
            if m:
                try:
                    stats[field] = float(m.group(1))
                except ValueError:
                    pass

        if stats:
            result[player_name.lower()] = stats

    # Strategy 2: if strategy 1 found nothing, try extracting flat {key:val} blocks
    if not result:
        for obj in re.findall(r'\{[^{}]{50,}\}', html):
            name_m = re.search(r'player_name["\s]*[:=]["\s]*"([^"]+)"', obj)
            if not name_m:
                continue
            player_name = name_m.group(1).strip()
            stats = {}
            for field in float_fields:
                m = re.search(
                    rf'[,\{{]?\s*"?{re.escape(field)}"?\s*:\s*(-?[\d]+(?:\.[\d]+)?)',
                    obj
                )
                if m:
                    try:
                        stats[field] = float(m.group(1))
                    except ValueError:
                        pass
            if stats:
                result[player_name.lower()] = stats

    print(f"  EPM: parsed {len(result)} players")

    if result:
        with open(cache_path, "w") as f:
            json.dump(result, f)
        print(f"  EPM: cached {len(result)} players")
    else:
        print("  EPM: no player stats parsed from HTML")

    return result


# ---------------------------------------------------------------------------
# Graph enrichment
# ---------------------------------------------------------------------------

def enrich_graph(
    graph: MatchupGraph,
    season: str = "2024-25",
    progress_callback=None,
) -> None:
    """
    Enrich every player node in the graph with:
      - Bio from CommonPlayerInfo (position, height, weight, team, …)
      - Per-game stats from PlayerCareerStats
      - Advanced stats from NBA API (off/def/net rating, USG%, PIE, AST%)
      - EPM + per-100 stats from dunksandthrees.com
    """
    # Fetch EPM data once for all players (single HTTP request, cached)
    season_year = int(season.split("-")[0]) + 1  # "2025-26" -> 2026
    epm_data = fetch_epm_data(season=season_year)

    total = len(graph.players)
    for i, (pid, player) in enumerate(graph.players.items()):
        if progress_callback:
            progress_callback(i, total, player.name)

        # Bio
        bio = get_player_bio(pid)
        if bio:
            player.position = bio.get("position") or player.position
            player.team = bio.get("team") or player.team
            player.height = bio.get("height") or player.height
            player.weight = bio.get("weight") or player.weight
            player.experience = bio.get("experience")
            player.jersey = bio.get("jersey")
            player.draft_year = bio.get("draft_year")
            player.draft_round = bio.get("draft_round")
            player.draft_pick = bio.get("draft_pick")

        # Per-game
        sg = get_player_season_stats(pid, season)
        if sg:
            player.ppg = sg.get("ppg")
            player.rpg = sg.get("rpg")
            player.apg = sg.get("apg")
            player.spg = sg.get("spg")
            player.bpg = sg.get("bpg")
            player.tov = sg.get("tov")
            player.mpg = sg.get("mpg")
            player.fg_pct = sg.get("fg_pct")
            player.fg3_pct = sg.get("fg3_pct")
            player.ft_pct = sg.get("ft_pct")
            player.ts_pct = sg.get("ts_pct")
            player.games = sg.get("games")

        # NBA API advanced
        adv = get_nba_advanced_stats(pid, season)
        if adv:
            player.off_rating = adv.get("off_rating")
            player.def_rating = adv.get("def_rating")
            player.net_rating = adv.get("net_rating")
            player.usg_pct = adv.get("usg_pct")
            player.pie = adv.get("pie")
            player.ast_pct = adv.get("ast_pct")
            if adv.get("ts_pct") is not None:
                player.ts_pct = adv.get("ts_pct")

        # EPM + per-100 from dunksandthrees.com (matched by lowercase name)
        epm = epm_data.get(player.name.lower(), {})
        if epm:
            player.epm_off = epm.get("off")
            player.epm_def = epm.get("def")
            player.epm_tot = epm.get("tot")
            player.p_pts_100 = epm.get("p_pts_100")
            player.p_ast_100 = epm.get("p_ast_100")
            player.p_blk_100 = epm.get("p_blk_100")
            player.p_stl_100 = epm.get("p_stl_100")
            player.p_drb_100 = epm.get("p_drb_100")
            player.p_orb_100 = epm.get("p_orb_100")
            player.p_tov_100 = epm.get("p_tov_100")
            player.p_fga_rim_100 = epm.get("p_fga_rim_100")
            player.p_fga_mid_100 = epm.get("p_fga_mid_100")
            player.p_fg3a_100 = epm.get("p_fg3a_100")
            player.p_fgpct_rim = epm.get("p_fgpct_rim")
            player.p_fgpct_mid = epm.get("p_fgpct_mid")

        # Update graph node attributes
        node_off = f"off_{pid}"
        node_def = f"def_{pid}"
        attrs = {
            "team": player.team,
            "position": player.position,
            "height": player.height,
            "ppg": player.ppg,
        }
        for node in (node_off, node_def):
            if node in graph.graph:
                graph.graph.nodes[node].update(attrs)


# ---------------------------------------------------------------------------
# Live team data — direct NBA.com API with nba_api fallback
# ---------------------------------------------------------------------------

def _fetch_team_stats_direct(season: str, force_refresh: bool) -> pd.DataFrame:
    """Try fetching team advanced stats directly from NBA.com stats API."""
    safe = season.replace("-", "_")
    cache_path = CACHE_DIR / f"team_stats_live_{safe}.json"
    # Pass params as a dict so requests handles URL encoding correctly
    session = _get_nba_session()
    if not force_refresh and cache_path.exists():
        try:
            with open(cache_path) as f:
                return _parse_nba_result_set(json.load(f))
        except Exception:
            pass
    params = {
        "Season": season,
        "SeasonType": "Regular Season",
        "MeasureType": "Advanced",
        "PerMode": "PerGame",
        "LeagueID": "00",
    }
    global _nba_session
    for attempt in range(2):
        try:
            resp = session.get(
                "https://stats.nba.com/stats/leaguedashteamstats",
                params=params,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            with open(cache_path, "w") as f:
                json.dump(data, f)
            return _parse_nba_result_set(data)
        except Exception as e:
            print(f"  Team stats direct fetch attempt {attempt + 1} failed: {e}")
            if attempt == 0:
                time.sleep(3)
                _nba_session = None
                session = _get_nba_session()
    return pd.DataFrame()


def _fetch_team_stats_nba_api(season: str) -> pd.DataFrame:
    """Fallback: fetch team stats via nba_api library (handles headers internally)."""
    try:
        base = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Base",
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
        )
        time.sleep(NBA_API_DELAY)
        base_df = base.get_data_frames()[0]

        adv = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
        )
        time.sleep(NBA_API_DELAY)
        adv_df = adv.get_data_frames()[0]

        adv_cols_to_drop = [c for c in adv_df.columns if c in base_df.columns and c != "TEAM_ID"]
        adv_df = adv_df.drop(columns=adv_cols_to_drop, errors="ignore")
        return base_df.merge(adv_df, on="TEAM_ID", how="left")
    except Exception as e:
        print(f"  Team stats nba_api fallback failed: {e}")
        return pd.DataFrame()


def get_team_stats_live(
    season: str = "2025-26",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch advanced team stats (Net Rtg, Off Rtg, Def Rtg, Pace, eFG%, TOV%,
    OReb%, TS%, PIE) for all 30 teams.

    Tries the direct NBA.com stats API first (params-encoded URL, no 500 errors);
    falls back to nba_api library if the direct call fails.
    """
    df = _fetch_team_stats_direct(season, force_refresh)
    if df.empty:
        print("  Falling back to nba_api for team stats…")
        df = _fetch_team_stats_nba_api(season)
    return df


def _fetch_standings_direct(season: str, force_refresh: bool) -> pd.DataFrame:
    """Try fetching standings directly from NBA.com leaguestandingsv3."""
    safe = season.replace("-", "_")
    cache_path = CACHE_DIR / f"standings_live_{safe}.json"
    session = _get_nba_session()
    if not force_refresh and cache_path.exists():
        try:
            with open(cache_path) as f:
                return _parse_nba_result_set(json.load(f))
        except Exception:
            pass
    params = {
        "LeagueID": "00",
        "Season": season,
        "SeasonType": "Regular Season",
    }
    global _nba_session
    for attempt in range(2):
        try:
            resp = session.get(
                "https://stats.nba.com/stats/leaguestandingsv3",
                params=params,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            with open(cache_path, "w") as f:
                json.dump(data, f)
            return _parse_nba_result_set(data)
        except Exception as e:
            print(f"  Standings direct fetch attempt {attempt + 1} failed: {e}")
            if attempt == 0:
                time.sleep(3)
                _nba_session = None
                session = _get_nba_session()
    return pd.DataFrame()


def _fetch_standings_nba_api(season: str) -> pd.DataFrame:
    """Fallback: fetch standings via nba_api library."""
    try:
        standings = leaguestandings.LeagueStandings(
            season=season,
            season_type="Regular Season",
        )
        time.sleep(NBA_API_DELAY)
        return standings.get_data_frames()[0]
    except Exception as e:
        print(f"  Standings nba_api fallback failed: {e}")
        return pd.DataFrame()


def get_team_standings_live(
    season: str = "2025-26",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch full league standings from leaguestandingsv3, including seeds,
    conference, win-loss, home/road splits, and clinch indicators.

    Tries direct NBA.com API first; falls back to nba_api on failure.
    Adds FULL_NAME = TeamCity + TeamName for cross-DataFrame matching.
    """
    df = _fetch_standings_direct(season, force_refresh)
    if df.empty:
        print("  Falling back to nba_api for standings…")
        df = _fetch_standings_nba_api(season)
    # Synthesise full team name for matching against team stats TEAM_NAME
    if "TeamCity" in df.columns and "TeamName" in df.columns:
        df["FULL_NAME"] = df["TeamCity"] + " " + df["TeamName"]
    elif "TEAM_CITY" in df.columns and "TEAM_NAME" in df.columns:
        df["FULL_NAME"] = df["TEAM_CITY"] + " " + df["TEAM_NAME"]
    return df


def get_playoff_series(
    season: str = "2025-26",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch the live playoff bracket / series state from commonplayoffseriesvector.

    Returns a DataFrame of all known playoff series with columns including
    HOME_TEAM_ID, VISITOR_TEAM_ID, HOME_SEED, VISITOR_SEED, ROUND, CONF,
    HOME_TEAM_WINS, VISITOR_TEAM_WINS, SERIES_STATUS, etc.
    Returns empty DataFrame if playoffs have not yet started.
    Cache: data/cache/playoff_series_{season}.json
    """
    safe = season.replace("-", "_")
    cache_path = CACHE_DIR / f"playoff_series_{safe}.json"
    url = (
        "https://stats.nba.com/stats/commonplayoffseriesvector"
        f"?Season={season}"
    )
    data = _fetch_nba_direct(url, cache_path, force_refresh)
    if not data:
        return pd.DataFrame()
    return _parse_nba_result_set(data)


def get_team_roster(
    team_id: int,
    season: str = "2025-26",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch the current roster for a team from commonteamroster.

    Returns a DataFrame with columns including PLAYER, PLAYER_ID, NUM, POSITION,
    HEIGHT, WEIGHT, BIRTH_DATE, AGE, EXP, SCHOOL, HOW_ACQUIRED.
    Cache: data/cache/roster_{team_id}_{season}.json
    """
    safe = season.replace("-", "_")
    cache_path = CACHE_DIR / f"roster_{team_id}_{safe}.json"
    url = (
        "https://stats.nba.com/stats/commonteamroster"
        f"?TeamID={team_id}&Season={season}"
    )
    data = _fetch_nba_direct(url, cache_path, force_refresh)
    if not data:
        return pd.DataFrame()
    # Result set 0 = roster, result set 1 = coaches
    return _parse_nba_result_set(data, idx=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val) -> Optional[float]:
    try:
        return float(val) if val is not None and val == val else None
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> Optional[int]:
    try:
        return int(val) if val is not None and val == val else None
    except (TypeError, ValueError):
        return None
