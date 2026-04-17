"""
CounterPoint — Narrative Drift Detection Engine

Detects when conventional scouting wisdom is statistically outdated.
For each player, computes z-score divergence between their current season
performance and their career baseline across key efficiency stats.

Flag types:
  "better_than_reputation"  — current numbers significantly above career avg
  "worse_than_reputation"   — current numbers significantly below career avg
  "role_shift"              — usage/assist indicators shifted materially
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import anthropic
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

DRIFT_THRESHOLD = 1.5          # z-score magnitude to trigger a flag
MIN_SEASONS_HISTORY = 2        # minimum prior seasons required
MIN_GAMES_PER_SEASON = 10      # ignore injury/cup-of-coffee rows

# Stats derived from SeasonTotalsRegularSeason per-game data
DRIFT_STATS = ["fg3_pct", "ts_pct", "efg_pct", "tov_pct", "ppg", "ast_pg", "ft_rate"]

# Basketball-relevance order for selecting the primary flag signal.
# The engine walks this list and picks the FIRST stat above DRIFT_THRESHOLD.
# FT Rate is a last resort — only flagged if |z| > FT_RATE_THRESHOLD.
STAT_PRIORITY = ["fg3_pct", "ts_pct", "efg_pct", "ppg", "ast_pg", "tov_pct", "ft_rate"]
FT_RATE_THRESHOLD = 3.0   # higher bar for FT rate to avoid noise

STAT_LABELS: Dict[str, str] = {
    "fg3_pct":  "3P%",
    "ts_pct":   "TS%",
    "efg_pct":  "eFG%",
    "tov_pct":  "TOV%",
    "ppg":      "PPG",
    "ast_pg":   "APG",
    "ft_rate":  "FT Rate",
}

# Stats where a lower value is better (affects flag direction)
LOWER_IS_BETTER = {"tov_pct"}

# Claude model to use for CounterPoint API calls
_CP_MODEL = "claude-sonnet-4-5"

# ── System prompts ─────────────────────────────────────────────────────────────

_BRIEFING_SYSTEM = (
    "You are an NBA front office analyst writing a pre-series intelligence briefing called CounterPoint. "
    "Your job is to identify players whose current statistical profile diverges from their established reputation. "
    "For each flagged player explain: (1) what the conventional scouting narrative says, "
    "(2) what the last 1-2 seasons of data actually show, and "
    "(3) the specific tactical implication for the opposing coaching staff. "
    "Be direct, specific, and cite the actual numbers provided. "
    "Never reference players not in the data provided."
)

_QA_SYSTEM = (
    "You are CounterPoint, an NBA analytics system. Answer using ONLY the data in the context. "
    "Rules you must follow without exception: "
    "(1) For questions about whether to guard, close out on, or load help toward a specific player — "
    "start your answer with YES or NO, then immediately cite the exact numbers: career average, "
    "prior 2 seasons, and current season value for the relevant stat. "
    "(2) Give the tactical implication in exactly one direct sentence. Pick one: "
    "perimeter coverage, help-side loading, pick-and-roll scheme adjustment, "
    "or whether last year's game plan is still valid. "
    "(3) If a player's drift score exceeds 2 standard deviations, say explicitly: "
    "'CounterPoint flags this as a materially outdated scouting narrative.' "
    "(4) If the data shows a player is STABLE (no drift flag), say so directly — "
    "cite their stable numbers and confirm the conventional scouting holds. "
    "Do not manufacture concern where the data shows none. "
    "(5) Never use general knowledge about a player if the provided data contradicts it."
)


# ── Raw-data helpers ───────────────────────────────────────────────────────────

def _safe_float(v) -> Optional[float]:
    try:
        return float(v) if v is not None and v == v else None
    except (TypeError, ValueError):
        return None


def _derive_season_stats(row: pd.Series) -> Dict:
    """Compute derived efficiency stats from a single career season row."""
    fgm  = _safe_float(row.get("FGM"))  or 0.0
    fga  = _safe_float(row.get("FGA"))  or 0.0
    fg3m = _safe_float(row.get("FG3M")) or 0.0
    fg3a = _safe_float(row.get("FG3A")) or 0.0
    fg3p = _safe_float(row.get("FG3_PCT"))
    fta  = _safe_float(row.get("FTA"))  or 0.0
    pts  = _safe_float(row.get("PTS"))
    tov  = _safe_float(row.get("TOV"))  or 0.0
    ast  = _safe_float(row.get("AST"))
    gp   = _safe_float(row.get("GP"))   or 0.0

    ts_denom = 2.0 * (fga + 0.44 * fta)
    ts_pct   = pts / ts_denom       if pts is not None and ts_denom > 0 else None
    efg_pct  = (fgm + 0.5 * fg3m) / fga  if fga > 2                   else None
    tov_d    = fga + 0.44 * fta + tov
    tov_pct  = tov / tov_d          if tov_d > 0                       else None
    ft_rate  = fta / fga             if fga > 2                        else None
    # Only count 3P% when the player has meaningful volume (≥1.5 attempts/game)
    fg3_pct  = fg3p                  if fg3a and fg3a >= 1.5           else None

    return {
        "season_id": str(row.get("SEASON_ID", "")),
        "team_abbr": str(row.get("TEAM_ABBREVIATION", "")),
        "gp":       gp,
        "fg3_pct":  fg3_pct,
        "ts_pct":   ts_pct,
        "efg_pct":  efg_pct,
        "tov_pct":  tov_pct,
        "ppg":      pts,
        "ast_pg":   ast,
        "ft_rate":  ft_rate,
    }


def _aggregate_career_splits(raw_df: pd.DataFrame) -> List[Dict]:
    """
    Collapse per-team rows into one row per season.
    Prefers the 'TOT' aggregate row (traded players); otherwise last row.
    Filters seasons with < MIN_GAMES_PER_SEASON.
    """
    season_stats: List[Dict] = []

    if "SEASON_ID" not in raw_df.columns:
        for _, row in raw_df.iterrows():
            s = _derive_season_stats(row)
            if s["gp"] >= MIN_GAMES_PER_SEASON:
                season_stats.append(s)
        return season_stats

    for season_id, group in raw_df.groupby("SEASON_ID", sort=False):
        if len(group) > 1 and "TEAM_ABBREVIATION" in group.columns:
            tot = group[group["TEAM_ABBREVIATION"] == "TOT"]
            row = tot.iloc[0] if not tot.empty else group.iloc[-1]
        else:
            row = group.iloc[-1]

        s = _derive_season_stats(row)
        if s["gp"] >= MIN_GAMES_PER_SEASON:
            season_stats.append(s)

    season_stats.sort(key=lambda x: x["season_id"])
    return season_stats


# ── Stat-priority flag selection ───────────────────────────────────────────────

def _select_flagged_stat(drift_scores: Dict[str, float]) -> Optional[str]:
    """
    Walk STAT_PRIORITY and return the first stat that clears DRIFT_THRESHOLD.
    FT Rate requires a higher threshold (FT_RATE_THRESHOLD) to suppress noise.
    Returns None if no stat qualifies — absence of a flag is meaningful.
    """
    for stat in STAT_PRIORITY[:-1]:       # all stats except ft_rate
        if stat in drift_scores and abs(drift_scores[stat]) >= DRIFT_THRESHOLD:
            return stat

    # FT Rate: last resort, higher bar
    if "ft_rate" in drift_scores and abs(drift_scores["ft_rate"]) >= FT_RATE_THRESHOLD:
        return "ft_rate"

    return None


def _flag_from_stat(stat: str, z: float) -> str:
    """Determine flag type from the selected stat and its z-score direction."""
    lower_better = stat in LOWER_IS_BETTER

    # Role shift: large movement in playmaking/usage proxy stats
    if stat == "ast_pg":
        return "role_shift"

    if (z > 0 and not lower_better) or (z < 0 and lower_better):
        return "better_than_reputation"
    return "worse_than_reputation"


# ── Player-specific narrative generation ──────────────────────────────────────

def _fmt_val(stat: str, v: float) -> str:
    fmt = ".1%" if "pct" in stat or stat == "ft_rate" else ".1f"
    return f"{v:{fmt}}"


def _narrative_texts(
    player_name: str,
    flag: str,
    stat: str,
    z: float,
    career_avgs: Dict,
    curr_vals: Dict,
) -> Tuple[str, str, str]:
    """
    Return (narrative, numbers_say, coaching_impl) — all player-specific.

    Each coaching_impl answers one of the four actionable questions:
      1. Should we guard this player differently on the perimeter?
      2. Should we load help toward or away from this player?
      3. Is this player more/less of a pick-and-roll threat?
      4. Has this player's role changed enough that last year's game plan is wrong?
    """
    last  = player_name.split()[-1]
    label = STAT_LABELS.get(stat, stat)
    ca    = career_avgs.get(stat)
    cv    = curr_vals.get(stat)
    ca_s  = _fmt_val(stat, ca) if ca is not None else "—"
    cv_s  = _fmt_val(stat, cv) if cv is not None else "—"
    abs_z = abs(z)

    # ── Three-point shooting ───────────────────────────────────────────────────
    if stat == "fg3_pct":
        if flag == "better_than_reputation":
            narrative = (
                f"{player_name}'s career 3P% of {ca_s} has not historically forced "
                f"tight close-out coverage — most defenses have been content to sag and "
                f"funnel them toward the paint."
            )
            numbers_say = (
                f"{last} is shooting {cv_s} from three this season — {abs_z:.1f} standard "
                f"deviations above their career mark of {ca_s}. That is not variance. "
                f"The perimeter threat is real."
            )
            coaching_impl = (
                f"Close out harder on {last} on the perimeter — drop coverage that worked "
                f"last season is now conceding open looks they are consistently converting."
            )
        else:
            narrative = (
                f"{player_name}'s career 3P% of {ca_s} has made them a perimeter coverage "
                f"priority — defenses have consistently dedicated close-out resources to "
                f"neutralizing their shooting."
            )
            numbers_say = (
                f"{last} is shooting only {cv_s} from three this season — {abs_z:.1f}σ "
                f"below their career average of {ca_s}. The three-point threat that drove "
                f"your close-out scheme is no longer there."
            )
            coaching_impl = (
                f"Sag off {last} on the perimeter and load the paint — their three-point "
                f"shooting this season does not justify the close-out resources your scheme "
                f"is currently spending on them."
            )

    # ── True shooting / eFG% ──────────────────────────────────────────────────
    elif stat in ("ts_pct", "efg_pct"):
        if flag == "better_than_reputation":
            narrative = (
                f"{player_name}'s career {label} of {ca_s} placed them in the capable-but-not-elite "
                f"efficiency tier — opponents have not historically needed to redirect major "
                f"defensive resources toward containing their scoring."
            )
            numbers_say = (
                f"{last} is converting at {cv_s} {label} this season — {abs_z:.1f}σ above "
                f"their career baseline of {ca_s}. They are a more dangerous offensive weapon "
                f"than your scouting history reflects."
            )
            coaching_impl = (
                f"Load help toward {last} — their shooting efficiency this season demands more "
                f"defensive attention than last year's game plan assigns."
            )
        else:
            narrative = (
                f"{player_name}'s career {label} of {ca_s} established them as a reliable "
                f"offensive threat — opponents have consistently prioritized containing their "
                f"scoring output."
            )
            numbers_say = (
                f"{last}'s {label} has dropped to {cv_s} this season — {abs_z:.1f}σ below "
                f"their career mark of {ca_s}. The shooting efficiency that justified elite "
                f"coverage is no longer there."
            )
            coaching_impl = (
                f"Redirect defensive attention away from {last} — their scoring efficiency "
                f"this season no longer merits the coverage load your scheme is committing."
            )

    # ── Scoring volume (PPG) ──────────────────────────────────────────────────
    elif stat == "ppg":
        if flag == "better_than_reputation":
            narrative = (
                f"{player_name} has averaged {ca_s} PPG for their career, placing them in "
                f"a secondary offensive role — not the primary threat defenses build their "
                f"game plan around."
            )
            numbers_say = (
                f"{last} is scoring {cv_s} PPG this season — {abs_z:.1f}σ above their "
                f"career norm of {ca_s}. They have become a primary offensive option that "
                f"your scouting files likely undervalue."
            )
            coaching_impl = (
                f"Commit more help-side attention toward {last} — the offensive volume "
                f"they are generating this season demands a higher defensive priority than "
                f"last season's scheme assigns."
            )
        else:
            narrative = (
                f"{player_name}'s career scoring average of {ca_s} PPG has made them a "
                f"featured offensive threat — defenses have built schemes around neutralizing "
                f"their scoring role."
            )
            numbers_say = (
                f"{last} is averaging only {cv_s} PPG this season — {abs_z:.1f}σ below "
                f"their career average of {ca_s}. The scoring output that drove defensive "
                f"priority has dropped materially."
            )
            coaching_impl = (
                f"Pull resources away from {last} and redistribute to higher-value coverage "
                f"assignments — their scoring this season does not justify the scheme "
                f"your team is currently running to contain them."
            )

    # ── Assist rate / playmaking (role shift) ─────────────────────────────────
    elif stat == "ast_pg":
        if z > 0:   # more assists = expanded playmaking role
            narrative = (
                f"{player_name}'s career average of {ca_s} APG placed them in a "
                f"scorer-first role — defenses have prepared for them primarily as a "
                f"finisher and shot-taker, not a shot creator."
            )
            numbers_say = (
                f"{last} is averaging {cv_s} assists per game this season — {abs_z:.1f}σ "
                f"above their career norm of {ca_s}. They have shifted into a primary "
                f"playmaking role your scouting files do not reflect."
            )
            coaching_impl = (
                f"Assign an active ball-hawker on {last} in pick-and-roll coverage — "
                f"they are operating as a shot creator this season, and last year's "
                f"scheme for them does not account for that volume."
            )
        else:       # fewer assists = contracted role
            narrative = (
                f"{player_name}'s career {ca_s} APG defined them as a primary creator "
                f"and playmaker — defenses have historically respected their ability "
                f"to generate shots for teammates."
            )
            numbers_say = (
                f"{last} is averaging only {cv_s} assists this season — {abs_z:.1f}σ "
                f"below their career norm of {ca_s}. The playmaking role that justified "
                f"complex defensive schemes has contracted."
            )
            coaching_impl = (
                f"Simplify your pick-and-roll coverage on {last} — they are not generating "
                f"the creation volume this season that historically required complex "
                f"defensive schemes to contain."
            )

    # ── Turnover rate ─────────────────────────────────────────────────────────
    elif stat == "tov_pct":
        if flag == "better_than_reputation":   # lower TOV% = better care of ball
            narrative = (
                f"{player_name}'s career turnover rate of {ca_s} has been a historically "
                f"exploitable weakness — aggressive pressure, trapping, and blitzes have "
                f"produced results against them."
            )
            numbers_say = (
                f"{last} is turning it over on only {cv_s} of possessions this season — "
                f"{abs_z:.1f}σ below their career rate of {ca_s}. They are handling "
                f"pressure far better than your game plan assumes."
            )
            coaching_impl = (
                f"Do not rely on turnover-forcing pressure against {last} — they are "
                f"taking better care of the ball this season, and trap-and-blitz schemes "
                f"built around forcing errors are not producing results."
            )
        else:   # higher TOV% = worse care of ball
            narrative = (
                f"{player_name}'s career TOV% of {ca_s} has not historically been an "
                f"exploitable weakness — defenses have not specifically targeted them "
                f"as a turnover risk."
            )
            numbers_say = (
                f"{last} is turning it over {cv_s} of possessions this season — "
                f"{abs_z:.1f}σ above their career rate of {ca_s}. They are a genuine "
                f"turnover risk your scheme should be targeting."
            )
            coaching_impl = (
                f"Attack {last} with trapping and ball pressure — their elevated turnover "
                f"rate this season is an exploitable weakness your defense has not "
                f"historically targeted."
            )

    # ── FT Rate (last resort) ─────────────────────────────────────────────────
    else:   # ft_rate
        if flag == "better_than_reputation":
            narrative = (
                f"{player_name}'s career FT rate of {ca_s} has established how aggressively "
                f"they seek contact — defenses have adjusted fouling habits around that pattern."
            )
            numbers_say = (
                f"{last} is drawing free throws at a {cv_s} rate this season — {abs_z:.1f}σ "
                f"above their career baseline of {ca_s}. They are attacking contact at "
                f"a materially higher rate."
            )
            coaching_impl = (
                f"Tighten foul discipline on {last} around the basket — their rate of "
                f"drawing contact has shifted enough to require tactical adjustment."
            )
        else:
            narrative = (
                f"{player_name}'s career FT rate of {ca_s} made drawing fouls a component "
                f"of their offensive game — defenses have been disciplined about foul "
                f"avoidance when guarding them."
            )
            numbers_say = (
                f"{last}'s FT rate has dropped to {cv_s} this season — {abs_z:.1f}σ below "
                f"their career baseline of {ca_s}. They are no longer generating free "
                f"throw opportunities at the rate opponents have prepared for."
            )
            coaching_impl = (
                f"Relax foul-avoidance schemes on {last} — they are not drawing contact "
                f"at the rate that historically justified defensive caution."
            )

    flag_desc = (
        f"{player_name}: {label} {ca_s} (career) vs {cv_s} (current), "
        f"{abs_z:.1f}σ drift — {FLAG_LABEL.get(flag, flag)}"
    )
    return narrative, numbers_say, coaching_impl, flag_desc


# ── Stable-stats summary (no-flag case) ───────────────────────────────────────

def _build_stable_summary(
    player_name: str,
    drift_scores: Dict,
    career_avgs: Dict,
    curr_vals: Dict,
) -> str:
    """
    Generate a one-sentence stable-stats confirmation for players below
    the drift threshold.  Picks the 2 most stable priority stats and
    reports their career vs. current values.
    """
    priority_available = [
        s for s in STAT_PRIORITY[:-1]
        if s in drift_scores and s in career_avgs and s in curr_vals
    ]
    if not priority_available:
        return f"{player_name} — insufficient career data for drift analysis."

    # Most stable = lowest |z-score|
    stable = sorted(priority_available, key=lambda s: abs(drift_scores.get(s, 0)))[:2]
    parts = []
    for s in stable:
        label = STAT_LABELS.get(s, s)
        ca = career_avgs[s]
        cv = curr_vals[s]
        parts.append(f"{label} (career: {_fmt_val(s, ca)}, current: {_fmt_val(s, cv)})")

    stat_str = " and ".join(parts)
    return (
        f"{player_name} — {stat_str} stable within career norms. "
        f"No coverage adjustment needed from last season's game plan."
    )


# ── Flag colour helpers ────────────────────────────────────────────────────────

FLAG_COLOR = {
    "better_than_reputation": "#00875A",
    "worse_than_reputation":  "#C8102E",
    "role_shift":             "#F0A500",
}

FLAG_LABEL = {
    "better_than_reputation": "Better than reputation",
    "worse_than_reputation":  "Worse than reputation",
    "role_shift":             "Role shift",
}


# ── Core drift engine ──────────────────────────────────────────────────────────

def compute_drift(
    player_id: int,
    raw_career_df: pd.DataFrame,
    current_season: str,
    player_name: str = "",
) -> Optional[Dict]:
    """
    Compute narrative drift for a single player.

    Args:
        player_id:       NBA player ID
        raw_career_df:   SeasonTotalsRegularSeason DataFrame from playercareerstats
        current_season:  Season string, e.g. "2025-26"
        player_name:     Display name for player-specific narrative text

    Returns:
        - dict with "flagged": True  + full drift data when a flag is triggered
        - dict with "flagged": False + stable_summary when below threshold
        - None only for truly missing / insufficient data
    """
    if raw_career_df.empty:
        return None

    season_stats = _aggregate_career_splits(raw_career_df)
    if not season_stats:
        return None

    curr_candidates = [s for s in season_stats if s["season_id"] == current_season]
    curr = curr_candidates[-1] if curr_candidates else season_stats[-1]
    curr_sid = curr["season_id"]

    history = [s for s in season_stats if s["season_id"] != curr_sid]
    if len(history) < MIN_SEASONS_HISTORY:
        return None

    prior_2 = history[-2:]

    drift_scores: Dict[str, float] = {}
    career_avgs:  Dict[str, float] = {}
    prior_2_vals: Dict[str, List[float]] = {}
    curr_vals:    Dict[str, float] = {}
    trajectories: Dict[str, Dict] = {}

    for stat in DRIFT_STATS:
        h_vals = [s[stat] for s in history if s.get(stat) is not None]
        c_val  = curr.get(stat)

        if len(h_vals) < 2 or c_val is None:
            continue

        mean = float(np.mean(h_vals))
        std  = float(np.std(h_vals))
        if std < 1e-6:
            continue

        z = (c_val - mean) / std
        drift_scores[stat] = float(z)
        career_avgs[stat]  = mean
        prior_2_vals[stat] = [float(s[stat]) for s in prior_2 if s.get(stat) is not None]
        curr_vals[stat]    = float(c_val)

        traj = season_stats[-5:] if len(season_stats) >= 5 else season_stats
        trajectories[stat] = {
            "seasons": [s["season_id"] for s in traj if s.get(stat) is not None],
            "values":  [float(s[stat])  for s in traj if s.get(stat) is not None],
        }

    if not drift_scores:
        return None

    # ── Priority-based stat selection (not raw max z-score) ───────────────────
    flagged_stat = _select_flagged_stat(drift_scores)

    if flagged_stat is None:
        # No stat crossed the threshold — return stable-profile dict
        return {
            "flagged":        False,
            "player_id":      player_id,
            "drift_scores":   drift_scores,
            "career_avgs":    career_avgs,
            "current_vals":   curr_vals,
            "stable_summary": _build_stable_summary(
                player_name, drift_scores, career_avgs, curr_vals
            ),
        }

    z_score = drift_scores[flagged_stat]
    flag    = _flag_from_stat(flagged_stat, z_score)

    narrative, numbers_say, coaching_impl, flag_desc = _narrative_texts(
        player_name, flag, flagged_stat, z_score, career_avgs, curr_vals
    )

    return {
        "flagged":         True,
        "player_id":       player_id,
        "drift_scores":    drift_scores,
        "max_drift_stat":  flagged_stat,
        "max_drift_score": z_score,
        "flag":            flag,
        "flag_desc":       flag_desc,
        "narrative":       narrative,
        "numbers_say":     numbers_say,
        "coaching_impl":   coaching_impl,
        "career_avgs":     career_avgs,
        "prior_2_vals":    prior_2_vals,
        "current_vals":    curr_vals,
        "trajectories":    trajectories,
        "seasons_analyzed": len(history) + 1,
        "stable_summary":  "",   # empty when flagged
    }


# ── Cross-team matchup extractor (deduplicated) ───────────────────────────────

def get_cross_team_matchups(
    graph,
    team1_name: str,
    team2_name: str,
    top_n: int = 10,
) -> List[Dict]:
    """
    Return the top-N cross-team matchup edges (by possessions) as dicts.
    Each player pair is only included once — whichever direction has more
    possessions wins; the reverse direction is skipped.
    """
    found = []
    seen_pairs: set = set()        # frozenset({off_pid, def_pid})

    t1w = set(w for w in team1_name.lower().split() if len(w) > 3)
    t2w = set(w for w in team2_name.lower().split() if len(w) > 3)

    # Sort by possessions descending so the higher-possession direction wins dedup
    sorted_matchups = sorted(
        graph.matchups.items(),
        key=lambda kv: kv[1].possessions,
        reverse=True,
    )

    for (off_id, def_id), edge in sorted_matchups:
        pair_key = frozenset({off_id, def_id})
        if pair_key in seen_pairs:
            continue

        off_p = graph.players.get(off_id)
        def_p = graph.players.get(def_id)
        if not off_p or not def_p:
            continue
        ot = (off_p.team or "").lower()
        dt = (def_p.team or "").lower()

        off_t1 = any(w in ot for w in t1w)
        off_t2 = any(w in ot for w in t2w)
        def_t1 = any(w in dt for w in t1w)
        def_t2 = any(w in dt for w in t2w)

        if (off_t1 and def_t2) or (off_t2 and def_t1):
            seen_pairs.add(pair_key)
            found.append({
                "off_player":  off_p.name,
                "off_pid":     off_id,
                "def_player":  def_p.name,
                "def_pid":     def_id,
                "off_team":    team1_name if off_t1 else team2_name,
                "def_team":    team2_name if def_t2 else team1_name,
                "ppp":         edge.points_per_possession,
                "possessions": edge.possessions,
                "fg_pct":      edge.fg_pct,
                "games":       edge.games_played,
            })
            if len(found) >= top_n:
                break

    return found


# ── Context formatters for Claude calls ───────────────────────────────────────

def _fmt_drift_ctx(player_name: str, drift: Dict) -> str:
    """Full stat-trajectory context for a flagged player."""
    lines = ["[COUNTERPOINT CONTEXT]", f"Player: {player_name}"]
    for stat, label in STAT_LABELS.items():
        if stat not in drift.get("career_avgs", {}):
            continue
        ca = drift["career_avgs"][stat]
        cv = drift["current_vals"].get(stat)
        p2 = drift.get("prior_2_vals", {}).get(stat, [])
        z  = drift["drift_scores"].get(stat, 0.0)
        lines.append(f"Career avg {label}: {_fmt_val(stat, ca)}")
        if len(p2) == 2:
            lines.append(
                f"Prior 2 seasons {label}: {_fmt_val(stat, p2[0])}, {_fmt_val(stat, p2[1])}"
            )
        elif len(p2) == 1:
            lines.append(f"Prior season {label}: {_fmt_val(stat, p2[0])}")
        if cv is not None:
            lines.append(f"Current season {label}: {_fmt_val(stat, cv)}")
        lines.append(f"Drift score ({label}): {z:+.2f}sigma")
    flag = drift.get("flag", "")
    flag_desc = drift.get("flag_desc", "")
    lines.append(f"Flag: {flag} -- {flag_desc}")
    return "\n".join(lines)


def _fmt_stable_ctx(player_name: str, drift: Dict) -> str:
    """Compact stat context for a player with no drift flag (stable profile)."""
    lines = [f"[STABLE PROFILE]", f"Player: {player_name}"]
    lines.append("Status: No significant narrative drift detected. Conventional scouting holds.")
    for stat, label in STAT_LABELS.items():
        ca = drift.get("career_avgs", {}).get(stat)
        cv = drift.get("current_vals", {}).get(stat)
        z  = drift.get("drift_scores", {}).get(stat, 0.0)
        if ca is not None and cv is not None:
            lines.append(
                f"{label}: career {_fmt_val(stat, ca)}, "
                f"current {_fmt_val(stat, cv)} (drift: {z:+.2f}sigma)"
            )
    return "\n".join(lines)


def _build_briefing_prompt(
    flagged: List[Dict],
    team1: str,
    team2: str,
    team1_stats: Optional[Dict],
    team2_stats: Optional[Dict],
) -> str:
    from llm_reports import _fmt_team_stats
    parts = [f"Matchup: {team1} vs {team2}\n"]
    if team1_stats:
        parts.append(_fmt_team_stats(team1, team1_stats))
    if team2_stats:
        parts.append(_fmt_team_stats(team2, team2_stats))
    parts.append("")
    for fp in flagged:
        parts.append(_fmt_drift_ctx(fp["name"], fp["drift"]))
        parts.append("")
    ctx = "\n".join(parts)
    return (
        f"Write a 2-3 paragraph pre-series CounterPoint intelligence briefing for "
        f"{team1} vs {team2} using the data below. Be direct and cite specific numbers.\n\n{ctx}"
    )


def _build_qa_prompt(
    user_query: str,
    graph,
    cp_state: Dict,
    team1: str,
    team2: str,
    team1_stats: Optional[Dict],
    team2_stats: Optional[Dict],
) -> str:
    from llm_reports import _fmt_team_stats
    ctx_parts: List[str] = []
    query_lower = user_query.lower()

    for pid, drift in cp_state.items():
        if drift is None:
            continue
        player = graph.players.get(pid) if graph else None
        if not player:
            continue
        # Match full name or last name
        name_lower = player.name.lower()
        last_lower = player.name.split()[-1].lower()
        if name_lower in query_lower or last_lower in query_lower:
            if drift.get("flagged"):
                ctx_parts.append(_fmt_drift_ctx(player.name, drift))
            else:
                ctx_parts.append(_fmt_stable_ctx(player.name, drift))

    for tm, stats in [(team1, team1_stats), (team2, team2_stats)]:
        if tm and stats and any(w in query_lower for w in tm.lower().split() if len(w) > 3):
            ctx_parts.append(_fmt_team_stats(tm, stats))

    # Default: inject all players in current cp_state
    if not ctx_parts:
        for pid, drift in cp_state.items():
            if drift is None:
                continue
            player = graph.players.get(pid) if graph else None
            if player:
                if drift.get("flagged"):
                    ctx_parts.append(_fmt_drift_ctx(player.name, drift))
                else:
                    ctx_parts.append(_fmt_stable_ctx(player.name, drift))
        if team1_stats:
            ctx_parts.append(_fmt_team_stats(team1, team1_stats))
        if team2_stats:
            ctx_parts.append(_fmt_team_stats(team2, team2_stats))

    ctx = "\n\n".join(ctx_parts) if ctx_parts else "(No CounterPoint data loaded for this matchup.)"
    return f"{ctx}\n\n[USER QUESTION]\n{user_query}"


# ── Claude API callers ─────────────────────────────────────────────────────────

def _sanitize(text: str) -> str:
    return (
        text
        .replace("\u2019", "'").replace("\u2018", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2013", "-").replace("\u2014", "-")
        .replace("\u2026", "...")
        .replace("\u03c3", "sigma")
        .encode("ascii", errors="replace").decode("ascii")
    )


def call_cp_briefing(
    flagged: List[Dict],
    team1: str,
    team2: str,
    team1_stats: Optional[Dict],
    team2_stats: Optional[Dict],
    api_key: str,
) -> str:
    prompt = _build_briefing_prompt(flagged, team1, team2, team1_stats, team2_stats)
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=_CP_MODEL,
            max_tokens=1400,
            system=_sanitize(_BRIEFING_SYSTEM),
            messages=[{"role": "user", "content": _sanitize(prompt)}],
        )
        return msg.content[0].text
    except anthropic.AuthenticationError:
        return "❌ Invalid Anthropic API key. Please check your key in the sidebar."
    except Exception as e:
        return f"❌ Briefing generation failed: {e}"


def call_cp_qa(
    user_query: str,
    graph,
    cp_state: Dict,
    team1: str,
    team2: str,
    team1_stats: Optional[Dict],
    team2_stats: Optional[Dict],
    chat_history: List[Dict],
    api_key: str,
) -> str:
    prompt = _build_qa_prompt(
        user_query, graph, cp_state, team1, team2, team1_stats, team2_stats
    )
    messages = list(chat_history[-6:])
    messages.append({"role": "user", "content": _sanitize(prompt)})
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=_CP_MODEL,
            max_tokens=900,
            system=_sanitize(_QA_SYSTEM),
            messages=messages,
        )
        return msg.content[0].text
    except anthropic.AuthenticationError:
        return "❌ Invalid Anthropic API key. Please check your key in the sidebar."
    except Exception as e:
        return f"❌ Query failed: {e}"


# ── Suggested question chips ───────────────────────────────────────────────────

def generate_example_questions(
    team1: str,
    team2: str,
    flagged_players: List[Dict],
) -> List[str]:
    """Return 3 dynamically generated example question strings."""
    questions: List[str] = []

    for fp in flagged_players[:2]:
        name  = fp["name"]
        last  = name.split()[-1]
        flag  = fp["drift"]["flag"]
        stat  = STAT_LABELS.get(fp["drift"]["max_drift_stat"], "stats")
        opp   = team2 if fp.get("off_team") == team1 else team1
        if flag == "better_than_reputation":
            questions.append(
                f"Should {opp.split()[-1]} guard {last} differently on the perimeter this series?"
            )
        elif flag == "worse_than_reputation":
            questions.append(
                f"Is {last}'s {stat} decline this season significant enough to change coverage?"
            )
        else:
            questions.append(f"Has {last}'s role shifted enough that last year's game plan is wrong?")

    while len(questions) < 3:
        if len(questions) == 0:
            questions.append(
                f"Who is the most misread player in the "
                f"{team1.split()[-1]} vs {team2.split()[-1]} series?"
            )
        elif len(questions) == 1:
            questions.append(
                f"Should the {team2.split()[-1]} load help or play straight-up defense?"
            )
        else:
            questions.append("Which flagged player should the defense prioritize adjusting to?")

    return questions[:3]
