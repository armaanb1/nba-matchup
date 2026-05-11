"""
Smart context builder for Ask the Analyst.

Detects teams, award concepts, and performance keywords in a question,
then resolves them to relevant player pools and fetches the right data
automatically — so the LLM always answers from real current stats,
not training memory.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

# ---------------------------------------------------------------------------
# Team name → abbreviation / graph lookup key
# ---------------------------------------------------------------------------

TEAM_ALIASES: Dict[str, str] = {
    # Full names + common short forms
    "atlanta hawks": "Hawks", "hawks": "Hawks",
    "boston celtics": "Celtics", "celtics": "Celtics",
    "brooklyn nets": "Nets", "nets": "Nets",
    "charlotte hornets": "Hornets", "hornets": "Hornets",
    "chicago bulls": "Bulls", "bulls": "Bulls",
    "cleveland cavaliers": "Cavaliers", "cavaliers": "Cavaliers", "cavs": "Cavaliers",
    "dallas mavericks": "Mavericks", "mavericks": "Mavericks", "mavs": "Mavericks",
    "denver nuggets": "Nuggets", "nuggets": "Nuggets",
    "detroit pistons": "Pistons", "pistons": "Pistons",
    "golden state warriors": "Warriors", "warriors": "Warriors", "golden state": "Warriors",
    "houston rockets": "Rockets", "rockets": "Rockets",
    "indiana pacers": "Pacers", "pacers": "Pacers",
    "la clippers": "Clippers", "los angeles clippers": "Clippers", "clippers": "Clippers",
    "la lakers": "Lakers", "los angeles lakers": "Lakers", "lakers": "Lakers",
    "memphis grizzlies": "Grizzlies", "grizzlies": "Grizzlies",
    "miami heat": "Heat", "heat": "Heat",
    "milwaukee bucks": "Bucks", "bucks": "Bucks",
    "minnesota timberwolves": "Timberwolves", "timberwolves": "Timberwolves", "wolves": "Timberwolves",
    "new orleans pelicans": "Pelicans", "pelicans": "Pelicans",
    "new york knicks": "Knicks", "knicks": "Knicks",
    "oklahoma city thunder": "Thunder", "oklahoma city": "Thunder", "okc": "Thunder", "thunder": "Thunder",
    "orlando magic": "Magic", "magic": "Magic",
    "philadelphia 76ers": "76ers", "76ers": "76ers", "sixers": "76ers",
    "phoenix suns": "Suns", "suns": "Suns",
    "portland trail blazers": "Trail Blazers", "trail blazers": "Trail Blazers", "blazers": "Trail Blazers",
    "sacramento kings": "Kings", "kings": "Kings",
    "san antonio spurs": "Spurs", "spurs": "Spurs",
    "toronto raptors": "Raptors", "raptors": "Raptors",
    "utah jazz": "Jazz", "jazz": "Jazz",
    "washington wizards": "Wizards", "wizards": "Wizards",
}

# ---------------------------------------------------------------------------
# Concept keyword → resolution strategy
# ---------------------------------------------------------------------------

CONCEPT_PATTERNS: List[Tuple[str, List[str]]] = [
    ("mvp",          ["mvp", "most valuable", "best player in the league"]),
    ("mip",          ["most improved", "mip", "biggest improvement", "stepped up", "breakout"]),
    ("dpoy",         ["defensive player of the year", "dpoy", "best defender", "lockdown"]),
    ("scoring_title",["scoring title", "leading scorer", "top scorer", "ppg leader"]),
    ("playoffs",     ["playoff", "playoffs", "postseason", "series", "first round", "second round",
                      "conference finals", "finals", "stepped up", "eliminated"]),
    ("sixth_man",    ["sixth man", "bench scorer", "off the bench", "bench player"]),
    ("rookie",       ["rookie", "first year", "first season", "roty", "rookie of the year"]),
    ("clutch",       ["clutch", "fourth quarter", "crunch time", "late game", "closing"]),
    ("efficiency",   ["efficient", "efficiency", "true shooting", "ts%", "best value"]),
]


def detect_teams(question: str) -> List[str]:
    """Return normalised team names mentioned in the question."""
    q = question.lower()
    found = []
    # Try longest matches first to avoid 'heat' matching 'oklahoma city thunder heat check'
    sorted_aliases = sorted(TEAM_ALIASES.keys(), key=len, reverse=True)
    for alias in sorted_aliases:
        if re.search(r'\b' + re.escape(alias) + r'\b', q):
            canonical = TEAM_ALIASES[alias]
            if canonical not in found:
                found.append(canonical)
    return found


def detect_concepts(question: str) -> List[str]:
    """Return concept keys mentioned in the question."""
    q = question.lower()
    found = []
    for concept, keywords in CONCEPT_PATTERNS:
        if any(kw in q for kw in keywords):
            if concept not in found:
                found.append(concept)
    return found


def get_team_players(team_name: str, graph, top_n: int = 8) -> List:
    """Return top players for a team from the loaded graph, by PPG."""
    players = [
        p for p in graph.players.values()
        if p.team and team_name.lower() in p.team.lower()
    ]
    players.sort(key=lambda p: p.ppg or 0, reverse=True)
    return players[:top_n]


def resolve_concept_players(concept: str, graph, career_dfs: Dict) -> List:
    """
    Return a list of Player objects relevant to a given concept.
    career_dfs: {player_id: career_df} for computing improvement scores.
    """
    all_players = list(graph.players.values())

    if concept == "mvp":
        # Top players by PPG + net rating
        candidates = [p for p in all_players if p.ppg and p.ppg >= 20]
        candidates.sort(key=lambda p: (p.ppg or 0) + (p.net_rating or 0) * 0.5, reverse=True)
        return candidates[:10]

    elif concept == "mip":
        # Players with biggest PPG improvement vs career weighted baseline
        scored = []
        for p in all_players:
            if not p.ppg:
                continue
            cdf = career_dfs.get(p.player_id)
            if cdf is None or cdf.empty or len(cdf) < 2:
                continue
            # Career baseline = weighted avg of all seasons except current
            past = cdf.iloc[1:]  # exclude most recent (current season)
            if past.empty:
                continue
            baseline_ppg = (past["ppg"] * past["weight"]).sum() / past["weight"].sum() if past["weight"].sum() > 0 else None
            if baseline_ppg is None or baseline_ppg == 0:
                continue
            improvement = (p.ppg - baseline_ppg) / baseline_ppg  # relative improvement
            scored.append((improvement, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:10]]

    elif concept == "dpoy":
        candidates = [p for p in all_players if p.def_rating is not None]
        candidates.sort(key=lambda p: (
            -(p.def_rating or 999) * 0.5 +
            (p.bpg or 0) * 2 +
            (p.spg or 0) * 2 +
            -(p.avg_ppp_def or 0.5) * 10
        ), reverse=True)
        return candidates[:8]

    elif concept == "scoring_title":
        candidates = [p for p in all_players if p.ppg]
        candidates.sort(key=lambda p: p.ppg or 0, reverse=True)
        return candidates[:8]

    elif concept == "sixth_man":
        # Low USG% but decent PPG (bench role proxy)
        candidates = [p for p in all_players if p.ppg and p.ppg >= 10 and p.usg_pct and p.usg_pct < 0.22]
        candidates.sort(key=lambda p: p.ppg or 0, reverse=True)
        return candidates[:6]

    elif concept == "clutch":
        candidates = [p for p in all_players if p.ppg and p.ppg >= 15]
        candidates.sort(key=lambda p: (p.ppg or 0) + (p.net_rating or 0) * 0.3, reverse=True)
        return candidates[:8]

    elif concept == "efficiency":
        candidates = [p for p in all_players if p.ts_pct and p.ppg and p.ppg >= 12]
        candidates.sort(key=lambda p: p.ts_pct or 0, reverse=True)
        return candidates[:8]

    elif concept == "playoffs":
        # High-usage, high-PPG players likely relevant to playoff questions
        candidates = [p for p in all_players if p.ppg and p.ppg >= 15]
        candidates.sort(key=lambda p: p.ppg or 0, reverse=True)
        return candidates[:10]

    return []


def fmt_player_compact(player, career_df: Optional[pd.DataFrame] = None) -> str:
    """One-line stat summary for a player — used for concept pools."""
    parts = [f"{player.name} ({player.team or '?'}, {player.position or '?'})"]
    stats = []
    if player.ppg: stats.append(f"PPG {player.ppg:.1f}")
    if player.rpg: stats.append(f"RPG {player.rpg:.1f}")
    if player.apg: stats.append(f"APG {player.apg:.1f}")
    if player.fg_pct: stats.append(f"FG% {player.fg_pct:.1%}")
    if player.ts_pct: stats.append(f"TS% {player.ts_pct:.1%}")
    if player.net_rating: stats.append(f"Net Rtg {player.net_rating:.1f}")
    if player.usg_pct: stats.append(f"USG% {player.usg_pct:.1%}")
    if player.avg_ppp_def: stats.append(f"PPP allowed {player.avg_ppp_def:.3f}")
    if stats:
        parts.append(" | ".join(stats))

    if career_df is not None and not career_df.empty and len(career_df) >= 2:
        curr = career_df.iloc[0]
        past = career_df.iloc[1:]
        if past["weight"].sum() > 0:
            b_ppg = (past["ppg"] * past["weight"]).sum() / past["weight"].sum()
            if b_ppg and b_ppg > 0 and curr.get("ppg"):
                delta = curr["ppg"] - b_ppg
                parts.append(f"PPG vs career baseline: {delta:+.1f} ({b_ppg:.1f} → {curr['ppg']:.1f})")

    return " | ".join(parts)
