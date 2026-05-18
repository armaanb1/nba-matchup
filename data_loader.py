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
from typing import Dict, List, Optional, Tuple

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


_FETCH_BACKOFF = [3, 8]          # seconds between retry attempts (attempt 1→2, 2→3)
_NEG_CACHE_TTL = 3600            # seconds before re-trying a previously-failed endpoint

def _neg_cache_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".failed")

def _neg_cache_fresh(cache_path: Path) -> bool:
    """True if a recent failure sentinel exists and hasn't expired."""
    p = _neg_cache_path(cache_path)
    if not p.exists():
        return False
    return (time.time() - p.stat().st_mtime) < _NEG_CACHE_TTL

def _write_neg_cache(cache_path: Path) -> None:
    try:
        _neg_cache_path(cache_path).touch()
    except Exception:
        pass

def _clear_neg_cache(cache_path: Path) -> None:
    try:
        _neg_cache_path(cache_path).unlink(missing_ok=True)
    except Exception:
        pass


def _fetch_nba_direct(
    url: str,
    cache_path: Path,
    force_refresh: bool = False,
    timeout: int = 15,
) -> Optional[Dict]:
    """
    Fetch a raw NBA.com stats API URL, with file-based caching.
    - Retries up to 3 times with exponential backoff (3 s, 8 s).
    - 503 responses get a longer first sleep (5 s) before retry.
    - After all retries fail with no stale cache, writes a negative-cache
      sentinel (.failed file) that suppresses re-fetching for 1 hour so the
      NBA API isn't hammered on every page load.
    - Falls back to stale cache if available.
    Returns parsed JSON or None on total failure.
    """
    if not force_refresh and cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            pass

    # Skip the network entirely if we recently confirmed this endpoint is down
    if not force_refresh and _neg_cache_fresh(cache_path):
        return None

    global _nba_session
    session = _get_nba_session()
    for attempt in range(3):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(data, f)
            _clear_neg_cache(cache_path)
            return data
        except requests.exceptions.TooManyRedirects:
            print(f"  Redirect loop for {url} — endpoint not available yet, returning None")
            return None
        except Exception as e:
            print(f"  Direct fetch attempt {attempt + 1} failed ({url}): {e}")
            if attempt < 2:
                # Longer sleep on 503 (server overloaded) vs other errors
                is_503 = "503" in str(e)
                sleep_secs = (5 if is_503 else _FETCH_BACKOFF[attempt])
                time.sleep(sleep_secs)
                _nba_session = None
                session = _get_nba_session()

    # All attempts failed — fall back to stale cache, then write negative sentinel
    if cache_path.exists():
        try:
            print(f"  Falling back to stale cache for {cache_path.name}")
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            pass
    _write_neg_cache(cache_path)
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
        if df.empty:
            cache_path.unlink()
            df = _fetch_matchup_data(season, season_type)
            if not df.empty:
                df.to_csv(cache_path, index=False)
    else:
        df = _fetch_matchup_data(season, season_type)
        if not df.empty:
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

    # Fields to extract — maps our attribute name to the site's JS key
    field_map = {
        "off": "off", "def": "def", "tot": "tot",
        "p_pts_100": "p_pts_100", "p_ast_100": "p_ast_100",
        "p_blk_100": "p_blk_100", "p_stl_100": "p_stl_100",
        "p_drb_100": "p_drb_100", "p_orb_100": "p_orb_100",
        "p_tov_100": "p_tov_100", "p_fga_rim_100": "p_fga_rim_100",
        "p_fga_mid_100": "p_fga_mid_100", "p_fg3a_100": "p_fg3a_100",
        "p_fgpct_rim": "p_fgpct_rim", "p_fgpct_mid": "p_fgpct_mid",
    }

    result = {}

    # The page embeds data as JS object literals with unquoted keys, e.g.:
    # {season:2026,player_id:1641705,player_name:"Victor Wembanyama",...,off:4.81,def:3.82,...}
    # Find each player_name and extract a window large enough to cover all fields
    for name_match in re.finditer(r'player_name:"([^"]+)"', html):
        player_name = name_match.group(1).strip()
        # Look back to find the opening brace, forward ~2000 chars for all fields
        chunk_start = max(0, name_match.start() - 300)
        chunk_end = min(len(html), name_match.end() + 2000)
        chunk = html[chunk_start:chunk_end]

        stats = {}
        for attr, js_key in field_map.items():
            # Match unquoted JS key: ,key:value or {key:value
            # Value can be integer, float, or null
            m = re.search(
                rf'(?<!["\w]){re.escape(js_key)}:(-?[0-9]+(?:\.[0-9]+)?)',
                chunk
            )
            if m:
                try:
                    stats[attr] = float(m.group(1))
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
    force_refresh_epm: bool = False,
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
    epm_data = fetch_epm_data(season=season_year, force_refresh=force_refresh_epm)

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
                timeout=15,
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
                timeout=15,
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


_MIN_GAMES_PER_SEASON = 10   # shared with counterpoint.py via counterpoint.MIN_GAMES_PER_SEASON


def _aggregate_career_splits_v2(raw_df: pd.DataFrame) -> List[Dict]:
    """
    Process SeasonTotalsRegularSeason DataFrame into structured per-season dicts.
    Prefers the TOT aggregate row for traded players.
    Filters seasons with < _MIN_GAMES_PER_SEASON games.
    """
    rows: List[Dict] = []
    if "SEASON_ID" not in raw_df.columns:
        return rows

    for season_id, group in raw_df.groupby("SEASON_ID", sort=False):
        if len(group) > 1 and "TEAM_ABBREVIATION" in group.columns:
            tot = group[group["TEAM_ABBREVIATION"] == "TOT"]
            row = tot.iloc[0] if not tot.empty else group.iloc[-1]
        else:
            row = group.iloc[-1]

        def _sf(col):
            v = row.get(col)
            try:
                return float(v) if v is not None and v == v else None
            except (TypeError, ValueError):
                return None

        gp   = _sf("GP") or 0.0
        if gp < _MIN_GAMES_PER_SEASON:
            continue

        fgm  = _sf("FGM") or 0.0
        fga  = _sf("FGA") or 0.0
        fg3m = _sf("FG3M") or 0.0
        fg3a = _sf("FG3A") or 0.0
        fg3p = _sf("FG3_PCT")
        fta  = _sf("FTA") or 0.0
        ftm  = _sf("FTM") or 0.0
        ft_pct_raw = _sf("FT_PCT")
        pts  = _sf("PTS")
        tov  = _sf("TOV") or 0.0
        ast  = _sf("AST")
        reb  = _sf("REB")

        ts_denom = 2.0 * (fga + 0.44 * fta)
        ts_pct   = pts / ts_denom       if pts is not None and ts_denom > 0 else None
        efg_pct  = (fgm + 0.5 * fg3m) / fga  if fga > 2                   else None
        tov_d    = fga + 0.44 * fta + tov
        tov_pct  = tov / tov_d          if tov_d > 0                       else None
        ft_rate  = fta / fga             if fga > 2                        else None
        ft_pct   = ft_pct_raw            if ft_pct_raw is not None         else (ftm / fta if fta > 0 else None)
        fg3_pct  = fg3p                  if fg3a and fg3a >= 1.5           else None
        # PerMode=PerGame — FTA and FG3A are already per-game averages
        fta_pg   = fta   if fta  > 0 else None
        fg3a_pg  = fg3a  if fg3a > 0 else None

        rows.append({
            "season_id": str(season_id),
            "team_abbr": str(row.get("TEAM_ABBREVIATION", "") or ""),
            "gp":        gp,
            "ppg":       pts,
            "rpg":       reb,
            "apg":       ast,
            "ast_pg":    ast,
            "fta_pg":    fta_pg,
            "fg3a_pg":   fg3a_pg,
            "fg3_pct":   fg3_pct,
            "ts_pct":    ts_pct,
            "efg_pct":   efg_pct,
            "tov_pct":   tov_pct,
            "ft_rate":   ft_rate,
            "ft_pct":    ft_pct,
        })
    return rows


def get_player_career_splits(
    player_id: int,
    force_refresh: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Fetch per-game regular-season career splits from playercareerstats.

    Returns a tuple (career_df, weighted_baseline) where:
      career_df         — one row per qualifying season (sorted descending, most recent first),
                          columns: season_id, team_abbr, gp, ppg, rpg, apg, ast_pg,
                                   fg3_pct, ts_pct, efg_pct, tov_pct, ft_rate, weight
      weighted_baseline — decay-weighted average for each tracked stat across all seasons.
                          Used by compute_drift as the comparison baseline.

    Exponential decay weights: weights[i] = 0.75**i (i=0 = most recent season).
    Raw API JSON cached at data/cache/career_splits_{player_id}.json.
    Processed result cached at data/cache/career_splits_processed_{player_id}.json.
    """
    proc_cache = CACHE_DIR / f"career_splits_processed_{player_id}.json"
    if not force_refresh and proc_cache.exists():
        try:
            with open(proc_cache) as f:
                cached = json.load(f)
            career_df = pd.DataFrame(cached["seasons"])
            return career_df, cached.get("weighted_baseline", {})
        except Exception:
            pass

    raw_cache = CACHE_DIR / f"career_splits_{player_id}.json"

    # Try nba_api endpoint first — handles NBA.com headers/rate-limits reliably
    raw_df = pd.DataFrame()
    try:
        career_ep = playercareerstats.PlayerCareerStats(
            player_id=player_id, per_mode_simple="PerGame", timeout=45
        )
        time.sleep(NBA_API_DELAY)
        raw_df = career_ep.get_data_frames()[0]
        # Persist raw response so future calls use file cache
        if not raw_df.empty and raw_cache:
            try:
                raw_cache.write_text(
                    career_ep.get_json(), encoding="utf-8"
                )
            except Exception:
                pass
    except Exception:
        pass

    # Fall back to direct URL fetch if nba_api failed
    if raw_df.empty:
        url = (
            "https://stats.nba.com/stats/playercareerstats"
            f"?PlayerID={player_id}&PerMode=PerGame"
        )
        data = _fetch_nba_direct(url, raw_cache, force_refresh=force_refresh, timeout=45)
        if not data:
            return pd.DataFrame(), {}
        raw_df = _parse_nba_result_set(data, idx=0)
    if raw_df.empty:
        return pd.DataFrame(), {}

    rows = _aggregate_career_splits_v2(raw_df)
    if not rows:
        return pd.DataFrame(), {}

    # Sort descending — most recent season first (index 0)
    rows.sort(key=lambda x: x["season_id"], reverse=True)

    # Exponential decay weights (index 0 = most recent = highest weight before norm)
    weights_raw = [0.75 ** i for i in range(len(rows))]
    w_sum = sum(weights_raw)
    weights_norm = [w / w_sum for w in weights_raw]
    for row, w in zip(rows, weights_norm):
        row["weight"] = w

    # Weighted baseline across all seasons for each tracked stat
    tracked = ["fg3_pct", "ts_pct", "efg_pct", "tov_pct", "ppg", "ast_pg", "ft_rate"]
    weighted_baseline: Dict = {}
    for stat in tracked:
        num = sum(r[stat] * r["weight"] for r in rows if r.get(stat) is not None)
        denom = sum(r["weight"] for r in rows if r.get(stat) is not None)
        if denom > 0:
            weighted_baseline[stat] = num / denom

    career_df = pd.DataFrame(rows)

    try:
        with open(proc_cache, "w") as f:
            json.dump({"seasons": rows, "weighted_baseline": weighted_baseline}, f)
    except Exception:
        pass

    return career_df, weighted_baseline


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


def get_team_head_coach(
    team_id: int,
    season: str = "2025-26",
    force_refresh: bool = False,
) -> Optional[str]:
    """
    Return the head coach name for a team, reusing the commonteamroster cache.
    Returns None if unavailable.
    """
    safe = season.replace("-", "_")
    cache_path = CACHE_DIR / f"roster_{team_id}_{safe}.json"
    url = (
        "https://stats.nba.com/stats/commonteamroster"
        f"?TeamID={team_id}&Season={season}"
    )
    data = _fetch_nba_direct(url, cache_path, force_refresh)
    if not data:
        return None
    coaches_df = _parse_nba_result_set(data, idx=1)
    if coaches_df.empty:
        return None
    # Head coach: IS_ASSISTANT == 0, or COACH_TYPE contains 'Head', or SORT_SEQUENCE == 1
    head = None
    for _, row in coaches_df.iterrows():
        is_asst = row.get("IS_ASSISTANT")
        coach_type = str(row.get("COACH_TYPE", "")).lower()
        sort_seq = row.get("SORT_SEQUENCE")
        if str(is_asst) == "0" or "head" in coach_type or sort_seq == 1:
            head = row.get("COACH_NAME") or f"{row.get('FIRST_NAME', '')} {row.get('LAST_NAME', '')}".strip()
            break
    return head or None


def get_team_h2h_record(
    team1_id: int,
    team2_id: int,
    season: str = "2025-26",
    season_type: str = "Regular Season",
) -> Dict:
    """
    Return the head-to-head record between two teams for the given season.

    Result dict:
      team1_wins, team2_wins, games: [{date, team1_score, team2_score, winner}]
    Cached at data/cache/h2h_{team1_id}_{team2_id}_{season}.json.
    """
    # Normalise so the smaller ID is always team1 in the cache key
    t_lo, t_hi = sorted([team1_id, team2_id])
    safe_season = season.replace("-", "_")
    cache_path = CACHE_DIR / f"h2h_{t_lo}_{t_hi}_{safe_season}.json"

    if cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            pass

    from nba_api.stats.endpoints import teamgamelog

    result: Dict = {"team1_wins": 0, "team2_wins": 0, "games": []}

    for tid, opp_id in [(team1_id, team2_id), (team2_id, team1_id)]:
        try:
            gl = teamgamelog.TeamGameLog(
                team_id=tid,
                season=season,
                season_type_all_star=season_type,
                timeout=30,
            )
            time.sleep(NBA_API_DELAY)
            df = gl.get_data_frames()[0]
            if df.empty:
                continue
            # Filter to games against the opponent
            opp_games = df[df["MATCHUP"].str.contains(
                _team_abbr_from_id(opp_id) or "XXXXXX", na=False
            )]
            for _, row in opp_games.iterrows():
                wl = str(row.get("WL", ""))
                pts = int(row.get("PTS", 0) or 0)
                # We'll fill in opponent score from the reverse pass
                result["games"].append({
                    "date": str(row.get("GAME_DATE", "")),
                    "team_id": tid,
                    "team_score": pts,
                    "wl": wl,
                })
                if wl == "W":
                    if tid == team1_id:
                        result["team1_wins"] += 1
                    else:
                        result["team2_wins"] += 1
        except Exception:
            pass

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f)
    except Exception:
        pass

    return result


def _team_abbr_from_id(team_id: int) -> Optional[str]:
    """Return the 3-letter abbreviation for a static team ID."""
    from nba_api.stats.static import teams as _st
    for t in _st.get_teams():
        if t["id"] == team_id:
            return t["abbreviation"]
    return None


def get_player_shot_chart(
    player_id: int,
    season: str = "2025-26",
    season_type: str = "Regular Season",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch shot-by-shot chart data for a player from shotchartdetail.

    Returns a DataFrame with columns including:
      LOC_X, LOC_Y, SHOT_ZONE_BASIC, SHOT_ZONE_AREA, SHOT_ZONE_RANGE,
      SHOT_MADE_FLAG, SHOT_DISTANCE, ACTION_TYPE, SHOT_TYPE

    Coordinates are in 10ths of feet; basket is at (0, 0).
    Cached to data/cache/shotchart_{player_id}_{season}_{safe_type}.json
    """
    safe_season = season.replace("-", "_")
    safe_type = season_type.replace(" ", "_")
    cache_path = CACHE_DIR / f"shotchart_{player_id}_{safe_season}_{safe_type}.json"

    season_type_enc = season_type.replace(" ", "+")
    url = (
        "https://stats.nba.com/stats/shotchartdetail"
        f"?PlayerID={player_id}"
        f"&Season={season}"
        f"&SeasonType={season_type_enc}"
        f"&ContextMeasure=FGA"
        f"&TeamID=0"
        f"&GameID="
        f"&LeagueID=00"
        f"&PlayerPosition="
        f"&DateFrom="
        f"&DateTo="
        f"&GameSegment="
        f"&LastNGames=0"
        f"&Location="
        f"&Month=0"
        f"&OpponentTeamID=0"
        f"&Outcome="
        f"&Period=0"
        f"&RookieYear="
        f"&VsConference="
        f"&VsDivision="
    )
    data = _fetch_nba_direct(url, cache_path, force_refresh=force_refresh)
    if not data:
        return pd.DataFrame()
    # resultSets[0] = shot chart data; resultSets[1] = league average zones
    return _parse_nba_result_set(data, idx=0)


def get_player_shot_zones(
    player_id: int,
    season: str = "2025-26",
    season_type: str = "Regular Season",
    force_refresh: bool = False,
) -> Dict[str, Dict]:
    """
    Aggregate shot chart data by SHOT_ZONE_BASIC.

    Returns a dict keyed by zone name:
      {zone: {"fga": int, "fgm": int, "pct": float, "freq": float}}
    where freq = fraction of total FGA from that zone.
    Returns {} if no data is available.
    """
    df = get_player_shot_chart(player_id, season, season_type, force_refresh)
    if df.empty or "SHOT_ZONE_BASIC" not in df.columns:
        return {}

    total_fga = len(df)
    if total_fga == 0:
        return {}

    result = {}
    for zone, group in df.groupby("SHOT_ZONE_BASIC"):
        fga = len(group)
        fgm = int(group["SHOT_MADE_FLAG"].sum()) if "SHOT_MADE_FLAG" in group.columns else 0
        result[zone] = {
            "fga": fga,
            "fgm": fgm,
            "pct": fgm / fga if fga > 0 else 0.0,
            "freq": fga / total_fga,
        }
    return result


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
