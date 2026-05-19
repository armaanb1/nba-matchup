"""
LLM Scouting Report generation using the Anthropic API.

The report synthesizes matchup graph data — edge weights, neighborhood context,
similar-player comparisons — into a natural-language scouting narrative.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import anthropic
import pandas as pd

from models import MatchupEdge, MatchupGraph, Player


# ---------------------------------------------------------------------------
# Context formatters
# ---------------------------------------------------------------------------

def _fmt_player_bio(player: Player) -> str:
    lines = [f"Name: {player.name}"]
    if player.position:
        lines.append(f"Position: {player.position}")
    if player.team:
        lines.append(f"Team: {player.team}")
    if player.height:
        lines.append(f"Height: {player.height}")
    if player.weight:
        lines.append(f"Weight: {player.weight} lbs")
    stats_parts = []
    for label, val in [("PPG", player.ppg), ("RPG", player.rpg), ("APG", player.apg),
                       ("SPG", player.spg), ("BPG", player.bpg)]:
        if val is not None:
            stats_parts.append(f"{label}: {val:.1f}")
    if player.fg_pct is not None:
        stats_parts.append(f"FG%: {player.fg_pct:.1%}")
    if player.ft_pct is not None:
        stats_parts.append(f"FT%: {player.ft_pct:.1%}")
    if player.ts_pct is not None:
        stats_parts.append(f"TS%: {player.ts_pct:.1%}")
    if stats_parts:
        lines.append("Season averages: " + ", ".join(stats_parts))
    adv_parts = []
    for label, val, fmt in [
        ("Off Rtg", player.off_rating, ".1f"),
        ("Def Rtg", player.def_rating, ".1f"),
        ("Net Rtg", player.net_rating, ".1f"),
        ("USG%", player.usg_pct, ".1%"),
        ("PIE", player.pie, ".3f"),
        ("AST%", player.ast_pct, ".1%"),
    ]:
        if val is not None:
            adv_parts.append(f"{label}: {val:{fmt}}")
    if adv_parts:
        lines.append("Advanced (NBA API): " + ", ".join(adv_parts))
    if player.avg_ppp_off is not None:
        lines.append(f"Avg PPP on offense this season: {player.avg_ppp_off:.3f}")
    if player.avg_ppp_def is not None:
        lines.append(f"Avg PPP allowed on defense this season: {player.avg_ppp_def:.3f}")
    return "\n".join(lines)


def _fmt_matchup(edge: MatchupEdge, off_name: str, def_name: str) -> str:
    lines = [
        f"Head-to-head: {off_name} (offense) vs {def_name} (defense)",
        f"  Possessions: {edge.possessions:.1f}  |  Games: {edge.games_played}",
        f"  Points: {edge.points:.1f}  |  PPP: {edge.points_per_possession:.3f}",
        f"  FG: {edge.fgm:.0f}/{edge.fga:.0f} ({edge.fg_pct:.1%})",
        f"  3PT: {edge.fg3m:.0f}/{edge.fg3a:.0f}",
        f"  eFG%: {edge.effective_fg_pct:.1%}",
        f"  AST: {edge.assists:.1f}  |  TOV: {edge.turnovers:.1f}  |  BLK: {edge.blocks:.1f}",
    ]
    return "\n".join(lines)


def _fmt_neighborhood_summary(rows: List[Dict], role: str, top_n: int = 5) -> str:
    if not rows:
        return "No neighborhood data available."
    ppp_key = "ppp" if role == "offense" else "ppp_allowed"
    opp_key = "defender" if role == "offense" else "scorer"
    team_key = "defender_team" if role == "offense" else "scorer_team"
    label = "best offensive matchups (highest PPP scored)" if role == "offense" \
        else "best defensive matchups (lowest PPP allowed)"

    # Filter out 0.000 PPP entries — these are data gaps, not real shutdowns
    valid = [r for r in rows if r[ppp_key] > 0]
    if not valid:
        return f"No valid {label} data available."

    best = sorted(valid, key=lambda x: x[ppp_key], reverse=(role == "offense"))[:top_n]
    # Exclude players already in best so lists are mutually exclusive
    best_set = {r[opp_key] for r in best}
    remaining = [r for r in valid if r[opp_key] not in best_set]
    worst = sorted(remaining, key=lambda x: x[ppp_key], reverse=(role != "offense"))[:top_n]

    arch_key = "defender_archetype" if role == "offense" else "scorer_archetype"

    def _fmt_row(r):
        team = r.get(team_key) or ""
        arch = r.get(arch_key) or ""
        meta = " · ".join(filter(None, [team, arch]))
        meta_str = f" ({meta})" if meta else ""
        return (
            f"  • vs {r[opp_key]}{meta_str}: PPP {r[ppp_key]:.3f} "
            f"FG% {r.get('fg_pct', r.get('fg_pct_allowed', 0)):.1%}  "
            f"({r['possessions']:.0f} poss)"
        )

    lines = [f"Top {top_n} {label}:"]
    for r in best:
        lines.append(_fmt_row(r))
    lines.append(f"\nBottom {top_n} (toughest matchups):")
    for r in worst:
        lines.append(_fmt_row(r))
    return "\n".join(lines)


def _fmt_shot_zones(zone_summary: Dict, player_name: str) -> str:
    """Format shot zone summary dict into LLM-readable context string."""
    if not zone_summary:
        return ""

    sorted_zones = sorted(zone_summary.items(), key=lambda x: x[1]["freq"], reverse=True)

    lines = [f"{player_name} shot distribution (FGA frequency + FG%):"]
    for zone, stats in sorted_zones:
        fga = stats["fga"]
        fgm = stats["fgm"]
        pct = stats["pct"]
        freq = stats["freq"]
        lines.append(
            f"  • {zone}: {freq:.0%} of FGA ({fgm}/{fga}, {pct:.1%} FG%)"
        )

    corner_3 = zone_summary.get("Left Corner 3", {})
    corner_3r = zone_summary.get("Right Corner 3", {})
    atb_3 = zone_summary.get("Above the Break 3", {})
    if (corner_3 or corner_3r) and atb_3:
        c3_fga = corner_3.get("fga", 0) + corner_3r.get("fga", 0)
        c3_fgm = corner_3.get("fgm", 0) + corner_3r.get("fgm", 0)
        c3_pct = c3_fgm / c3_fga if c3_fga > 0 else 0
        atb_pct = atb_3.get("pct", 0)
        lines.append(
            f"  ↳ Corner 3 combined: {c3_fga} FGA at {c3_pct:.1%} — "
            f"Above-break 3: {atb_3.get('fga', 0)} FGA at {atb_pct:.1%}"
        )

    return "\n".join(lines)


def fmt_career_context(career_df: pd.DataFrame, player_name: str, shot_zones: Optional[Dict] = None) -> str:
    """Format career splits + optional current-season shot zones for LLM context."""
    result = _fmt_career_trajectory(career_df, player_name)
    if shot_zones:
        zone_text = _fmt_shot_zones(shot_zones, player_name)
        if zone_text:
            if result:
                result = result + "\n\n" + zone_text
            else:
                result = zone_text
    return result


def fmt_current_season_context(
    player: "Player",
    shot_zones: Optional[Dict] = None,
    off_neighborhood: Optional[List[Dict]] = None,
    def_neighborhood: Optional[List[Dict]] = None,
) -> str:
    """Format a player's current-season enriched stats for use as LLM context."""
    bio = _fmt_player_bio(player)
    zones = _fmt_shot_zones(shot_zones or {}, player.name) if shot_zones else ""
    parts = [f"=== {player.name.upper()} — THIS SEASON (enriched) ===", bio]
    if zones:
        parts.append(zones)
    if off_neighborhood:
        parts.append(_fmt_neighborhood_summary(off_neighborhood, role="offense", top_n=6))
    if def_neighborhood:
        parts.append(_fmt_neighborhood_summary(def_neighborhood, role="defense", top_n=6))
    return "\n".join(parts)


def _fmt_career_trajectory(career_df: pd.DataFrame, player_name: str) -> str:
    """
    Format the multi-season career DataFrame into a compact LLM-readable block.
    Returns empty string if career_df is empty, has fewer than 2 rows, or on any error.
    """
    try:
        if career_df is None or career_df.empty or len(career_df) < 2:
            return ""

        import math

        def _ok(v) -> bool:
            if v is None:
                return False
            try:
                return math.isfinite(float(v))
            except (TypeError, ValueError):
                return False

        def _pct(v) -> str:
            return f"{float(v):.0%}" if _ok(v) else "—"

        def _f1(v) -> str:
            return f"{float(v):.1f}" if _ok(v) else "—"

        curr = career_df.iloc[0]
        curr_sid = str(curr.get("season_id", "current season"))

        # Prominent current-season callout at the top
        lines = [
            f"{player_name} career trajectory (most recent first):",
            f"*** CURRENT SEASON ({curr_sid}): "
            f"PPG {_f1(curr.get('ppg'))} | "
            f"3P% {_pct(curr.get('fg3_pct'))} | "
            f"TS% {_pct(curr.get('ts_pct'))} | "
            f"APG {_f1(curr.get('apg') if _ok(curr.get('apg')) else curr.get('ast_pg'))} | "
            f"FTA/g {_f1(curr.get('fta_pg'))} | "
            f"FT% {_pct(curr.get('ft_pct'))} | "
            f"3PA/g {_f1(curr.get('fg3a_pg'))} ***",
            f"(Use this season as the primary reference. Only cite older seasons to show a trend.)",
            "",
            f"{'Season':<9} {'GP':>4} {'PPG':>6} {'3P%':>5} {'TS%':>5} {'FT%':>5} {'FTA/g':>6} {'3PA/g':>6} {'APG':>5}",
        ]

        for _, row in career_df.iterrows():
            sid = str(row.get("season_id", ""))
            gp_raw = row.get("gp")
            gp   = int(float(gp_raw)) if _ok(gp_raw) else 0
            ppg  = _f1(row.get("ppg"))
            f3   = _pct(row.get("fg3_pct"))
            ts   = _pct(row.get("ts_pct"))
            ft   = _pct(row.get("ft_pct"))
            fta  = _f1(row.get("fta_pg"))
            f3a  = _f1(row.get("fg3a_pg"))
            apg_raw = row.get("apg") if _ok(row.get("apg")) else row.get("ast_pg")
            apg  = _f1(apg_raw)
            lines.append(
                f"{sid:<9} {gp:>4} {ppg:>6} {f3:>5} {ts:>5} {ft:>5} {fta:>6} {f3a:>6} {apg:>5}"
            )

        def _wbase(col):
            col_vals = []
            for _, r in career_df.iterrows():
                v = r.get(col)
                wt = r.get("weight", 0)
                if _ok(v) and _ok(wt):
                    col_vals.append((float(v), float(wt)))
            if not col_vals:
                return None
            num   = sum(v * wt for v, wt in col_vals)
            denom = sum(wt for _, wt in col_vals)
            result = num / denom if denom else None
            return result if _ok(result) else None

        b_ppg  = _wbase("ppg")
        b_f3   = _wbase("fg3_pct")
        b_ts   = _wbase("ts_pct")
        b_ft   = _wbase("ft_pct")
        b_apg  = _wbase("ast_pg")
        b_fta  = _wbase("fta_pg")
        b_f3a  = _wbase("fg3a_pg")

        lines.append(
            f"Weighted career baseline: PPG {_f1(b_ppg)} | 3P% {_pct(b_f3)} | "
            f"TS% {_pct(b_ts)} | FT% {_pct(b_ft)} | FTA/g {_f1(b_fta)} | 3PA/g {_f1(b_f3a)} | APG {_f1(b_apg)}"
        )

        def _delta_pct(cur, base) -> str:
            if not _ok(cur) or not _ok(base):
                return "—"
            d = float(cur) - float(base)
            return f"{d:+.0%}" if _ok(d) else "—"

        def _delta_f(cur, base) -> str:
            if not _ok(cur) or not _ok(base):
                return "—"
            d = float(cur) - float(base)
            return f"{d:+.1f}" if _ok(d) else "—"

        lines.append(
            f"Current vs baseline: "
            f"PPG {_delta_f(curr.get('ppg'), b_ppg)} | "
            f"3P% {_delta_pct(curr.get('fg3_pct'), b_f3)} | "
            f"TS% {_delta_pct(curr.get('ts_pct'), b_ts)} | "
            f"FT% {_delta_pct(curr.get('ft_pct'), b_ft)} | "
            f"FTA/g {_delta_f(curr.get('fta_pg'), b_fta)} | "
            f"3PA/g {_delta_f(curr.get('fg3a_pg'), b_f3a)} | "
            f"APG {_delta_f(curr.get('ast_pg'), b_apg)}"
        )

        return "\n".join(lines)

    except Exception:
        return ""


def _fmt_similar_defenders(similar_list: List[Dict], top_n: int = 5) -> str:
    if not similar_list:
        return "No similar defenders found."
    lines = ["Defenders with most similar matchup profile:"]
    for s in similar_list[:top_n]:
        lines.append(
            f"  • {s['defender']} ({s.get('team','')}, {s.get('position','')}): "
            f"similarity={s['combined_score']:.2f}, shared opponents={s['shared_opponents']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are writing a matchup analysis column. Your job is to explain why "
    "certain offensive players thrive against certain defensive archetypes — "
    "and what that means for how teams should build rosters. Every report ends "
    "with a conclusion: a specific schematic adjustment, a defensive assignment, "
    "or a roster move. Never summarize the data. Deliver a verdict."

    "\n\nVoice rules — non-negotiable:"

    "\n1. Never lead a sentence with a stat. Stats belong parenthetically inside "
    "the argument. "
    "Wrong: '46.7% FG and 0.298 PPP this season.' "
    "Right: 'Brunson lives in the mid-range and the paint — efficiently "
    "(46.7% FG, 0.298 PPP) — because his frame works in his favor there.' "
    "Only use numbers present in the data provided. Never invent or approximate "
    "a statistic."

    "\n2. Every cited number needs a mechanical explanation. What physical "
    "attribute, skill, or scheme produced it? If you cannot explain the why, "
    "do not cite the number."

    "\n3. Archetypes over individuals. When describing who a player beats or "
    "struggles against, lead with the archetype — the physical or schematic "
    "type that causes the outcome — then name specific players as examples."

    "\n4. The head-to-head section is the sharpest part of any matchup report. "
    "Open with the headline finding — synthesize PPP, FG%, possessions, and "
    "turnovers into one clear picture. Then give the physical reason it happened. "
    "Then state the game-plan implication. No setup. Get to the finding "
    "immediately."

    "\n5. The strategic recommendation must be a specific action with a specific "
    "trigger: name the play type, the coverage scheme, and the floor zone to "
    "force the offensive player toward. Reference the offensive player's weakest "
    "high-volume zone and the defender's best physical tool. No generic advice."

    "\n6. Use scout language: paint touches, leverage points, help rotations, "
    "verticality, shade coverage, screen navigation, drop vs. hedge vs. switch. "
    "Never write: 'it is worth noting', 'demonstrates', 'exhibits', 'showcases', "
    "'certainly', 'great question', or 'as an AI.'"

    "\n7. Career trajectory data: only reference a trend when it changes the "
    "tactical read. If a player's 3P% has dropped three straight seasons, mention "
    "it when discussing how hard to chase them off the line. If the trend "
    "confirms what this season shows, skip it. Current season stats always lead."

    "\n8. No titles, headers, paragraph labels, or bullet points anywhere in the "
    "report. Continuous prose only. Every report ends with a verdict the reader "
    "can act on: a roster move, a coverage scheme, or a defensive assignment. "
    "Never end with a summary of what was already said."
)


def _fmt_team_context(
    off_team: str,
    def_team: str,
    off_record: Optional[Dict] = None,
    def_record: Optional[Dict] = None,
    h2h: Optional[Dict] = None,
) -> str:
    """Format team records and H2H into a context block."""
    lines = []
    if off_record:
        w, l = int(off_record.get("W", 0) or 0), int(off_record.get("L", 0) or 0)
        conf = off_record.get("Conference", "")
        seed = off_record.get("PlayoffRank")
        seed_str = f", {seed}{_ordinal(seed)} seed" if seed and seed < 99 else ""
        lines.append(f"{off_team}: {w}-{l}{(' (' + conf + seed_str + ')') if conf else ''}")
    if def_record:
        w, l = int(def_record.get("W", 0) or 0), int(def_record.get("L", 0) or 0)
        conf = def_record.get("Conference", "")
        seed = def_record.get("PlayoffRank")
        seed_str = f", {seed}{_ordinal(seed)} seed" if seed and seed < 99 else ""
        lines.append(f"{def_team}: {w}-{l}{(' (' + conf + seed_str + ')') if conf else ''}")
    if h2h and (h2h.get("team1_wins", 0) + h2h.get("team2_wins", 0)) > 0:
        t1w = h2h["team1_wins"]
        t2w = h2h["team2_wins"]
        total = t1w + t2w
        lines.append(
            f"Season series ({total}g): {off_team} {t1w}-{t2w} {def_team}"
        )
    return "\n".join(lines)


def _ordinal(n) -> str:
    try:
        n = int(n)
        return {1: "st", 2: "nd", 3: "rd"}.get(n if n < 20 else n % 10, "th")
    except (TypeError, ValueError):
        return "th"


def _build_matchup_prompt(
    edge: MatchupEdge,
    off_player: Player,
    def_player: Player,
    off_neighborhood: List[Dict],
    def_neighborhood: List[Dict],
    off_shot_zones: Optional[Dict] = None,
    def_shot_zones: Optional[Dict] = None,
    off_career_df: Optional[pd.DataFrame] = None,
    def_career_df: Optional[pd.DataFrame] = None,
    off_team_record: Optional[Dict] = None,
    def_team_record: Optional[Dict] = None,
    h2h_record: Optional[Dict] = None,
) -> str:
    """Build the matchup report prompt string (shared by sync and streaming callers)."""
    off_zone_ctx = _fmt_shot_zones(off_shot_zones or {}, off_player.name)
    def_zone_ctx = _fmt_shot_zones(def_shot_zones or {}, def_player.name)

    zone_section = ""
    if off_zone_ctx or def_zone_ctx:
        zone_section = "\n\n=== SHOT DISTRIBUTION ==="
        if off_zone_ctx:
            zone_section += f"\n{off_zone_ctx}"
        if def_zone_ctx:
            zone_section += f"\n\n{def_zone_ctx}"

    off_career_ctx = _fmt_career_trajectory(off_career_df, off_player.name) if off_career_df is not None and not off_career_df.empty else ""
    def_career_ctx = _fmt_career_trajectory(def_career_df, def_player.name) if def_career_df is not None and not def_career_df.empty else ""

    off_career_section = f"\n\n=== {off_player.name.upper()} CAREER TRAJECTORY ===\n{off_career_ctx}" if off_career_ctx else ""
    def_career_section = f"\n\n=== {def_player.name.upper()} CAREER TRAJECTORY ===\n{def_career_ctx}" if def_career_ctx else ""

    team_ctx = _fmt_team_context(
        off_player.team or "Unknown", def_player.team or "Unknown",
        off_team_record, def_team_record, h2h_record,
    )
    team_section = f"\n\n=== SERIES CONTEXT ===\n{team_ctx}" if team_ctx else ""

    context = f"""
=== OFFENSIVE PLAYER: {off_player.name.upper()} ===
{_fmt_player_bio(off_player)}{off_career_section}

=== DEFENSIVE PLAYER: {def_player.name.upper()} ===
{_fmt_player_bio(def_player)}{def_career_section}

=== HEAD-TO-HEAD MATCHUP ===
{_fmt_matchup(edge, off_player.name, def_player.name)}

=== {off_player.name.upper()} OFFENSIVE CONTEXT ===
{_fmt_neighborhood_summary(off_neighborhood, role='offense')}

=== {def_player.name.upper()} DEFENSIVE CONTEXT ===
{_fmt_neighborhood_summary(def_neighborhood, role='defense')}{zone_section}{team_section}
""".strip()

    return (
        f"Write a matchup analysis on {off_player.name} being guarded by "
        f"{def_player.name}. Two paragraphs only.\n\n"
        f"Paragraph 1 — THE MATCHUP: Open with the headline finding from the "
        f"head-to-head data — synthesize PPP, FG%, possessions, and turnovers into "
        f"one clear picture of how this pairing played out. Then give the physical "
        f"reason it happened. Use the offensive and defensive archetype context to "
        f"explain whether the result was expected — does {off_player.name} typically "
        f"feast on or struggle against {def_player.name}'s archetype, and did the "
        f"data confirm or break that trend? If the result defied the expected archetype "
        f"matchup, explain why. If it confirmed it, state what physical tool or "
        f"schematic trait made the difference.\n\n"
        f"Paragraph 2 — THE VERDICT: One specific triggered action. Name the play "
        f"type, the coverage scheme, and the floor zone to force {off_player.name} "
        f"toward. Reference their weakest high-volume zone and {def_player.name}'s "
        f"best physical tool. If the archetype matchup favors the offense, the verdict "
        f"must reflect urgency — what the defense must do differently. If it favors "
        f"the defense, state what the offense needs to counter. No generic advice.\n\n"
        f"Only reference {def_player.name}'s offensive role if their scoring load is "
        f"high enough to force opponent trade-offs in how they load toward their "
        f"defensive assignment. Use stats parenthetically as evidence throughout.\n\n"
        f"{context}"
    )


def generate_matchup_report(
    edge: MatchupEdge,
    off_player: Player,
    def_player: Player,
    off_neighborhood: List[Dict],
    def_neighborhood: List[Dict],
    api_key: str,
    off_shot_zones: Optional[Dict] = None,
    def_shot_zones: Optional[Dict] = None,
    off_career_df: Optional[pd.DataFrame] = None,
    def_career_df: Optional[pd.DataFrame] = None,
    off_team_record: Optional[Dict] = None,
    def_team_record: Optional[Dict] = None,
    h2h_record: Optional[Dict] = None,
) -> str:
    """Generate a scouting report (blocking, returns full string)."""
    prompt = _build_matchup_prompt(
        edge, off_player, def_player, off_neighborhood, def_neighborhood,
        off_shot_zones, def_shot_zones, off_career_df, def_career_df,
        off_team_record, def_team_record, h2h_record,
    )
    return _call_anthropic(prompt, api_key)


def stream_matchup_report(
    edge: MatchupEdge,
    off_player: Player,
    def_player: Player,
    off_neighborhood: List[Dict],
    def_neighborhood: List[Dict],
    api_key: str,
    off_shot_zones: Optional[Dict] = None,
    def_shot_zones: Optional[Dict] = None,
    off_career_df: Optional[pd.DataFrame] = None,
    def_career_df: Optional[pd.DataFrame] = None,
    off_team_record: Optional[Dict] = None,
    def_team_record: Optional[Dict] = None,
    h2h_record: Optional[Dict] = None,
):
    """Streaming variant — yields text chunks as they arrive from the API."""
    prompt = _build_matchup_prompt(
        edge, off_player, def_player, off_neighborhood, def_neighborhood,
        off_shot_zones, def_shot_zones, off_career_df, def_career_df,
        off_team_record, def_team_record, h2h_record,
    )
    return stream_report(prompt, api_key)


def _build_profile_prompt(
    player: Player,
    role: str,
    neighborhood: List[Dict],
    shot_zones: Optional[Dict] = None,
    career_df: Optional[pd.DataFrame] = None,
) -> str:
    """Build the player profile report prompt string."""
    zone_ctx = _fmt_shot_zones(shot_zones or {}, player.name)
    zone_section = f"\n\n=== SHOT DISTRIBUTION ===\n{zone_ctx}" if zone_ctx else ""

    career_ctx = _fmt_career_trajectory(career_df, player.name) if career_df is not None and not career_df.empty else ""
    career_section = f"\n\n=== CAREER TRAJECTORY ===\n{career_ctx}" if career_ctx else ""

    context = f"""
=== PLAYER PROFILE: {player.name.upper()} ===
{_fmt_player_bio(player)}{career_section}

=== MATCHUP NEIGHBORHOOD ({role.upper()}) ===
{_fmt_neighborhood_summary(neighborhood, role=role, top_n=6)}{zone_section}
""".strip()

    if role == "offense":
        return (
            f"Write a scouting report on {player.name} as an offensive player, for a "
            f"coaching staff preparing to defend them. Four paragraphs.\n\n"
            f"Paragraph 1 — OFFENSIVE PROFILE: Name the offensive archetype. Where does "
            f"{player.name} score and why do those zones work given their physical tools. "
            f"Distinguish which zones are genuine weapons vs. which they operate in at "
            f"high volume without efficiency. If shot zone data is provided, use it to "
            f"draw conclusions about how a defense must load and position — do not list "
            f"the zones. Address how central they are to the offense using USG% "
            f"contextualized against the rest of the league — if they are particularly "
            f"high or low, say so and explain the implication. Note whether they are a "
            f"meaningful playmaker, an offensive rebounder, or a liability in any area — "
            f"including negative indicators like TOV% or inefficient high-volume zones. "
            f"Use percentile context to show where they stand relative to the league, "
            f"especially where they are an outlier.\n\n"
            f"Paragraph 2 — DEFENSIVE ARCHETYPE ANALYSIS: Which defensive archetypes "
            f"contain them and the mechanical reason why those players disrupt their game. "
            f"Which archetypes get exploited and why. Use the neighborhood data to "
            f"identify which defenders have held them down and which have gotten torched — "
            f"explain the physical reason behind each pattern. Do not rely on PPP alone — "
            f"also use field goal percentage, efficiency, and turnovers to fully "
            f"contextualize those results.\n\n"
            f"Paragraph 3 — TENDENCY AND TRAJECTORY: If career trajectory data shows a "
            f"trend that changes the defensive read — declining FTA, three-point volume "
            f"without efficiency, slipping rim conversion, role shift — state it and "
            f"explain what it means for how much to respect each part of their game. If "
            f"the trend confirms what this season already shows, skip it. Do not write a "
            f"career section.\n\n"
            f"Paragraph 4 — THE VERDICT: One specific scheme recommendation. Name the "
            f"play type, the coverage scheme, and the floor zone to force {player.name} "
            f"toward. Every number in this report supports an argument about how to guard "
            f"them.\n\n{context}"
        )
    else:
        return (
            f"Write a scouting report on {player.name} as a defender, for a "
            f"coaching staff preparing to attack them. Four paragraphs.\n\n"
            f"Paragraph 1 — DEFENSIVE ARCHETYPE: Name the defensive role. What physical "
            f"and schematic traits define how {player.name} guards. What are their best "
            f"tools and where do they apply on the floor. Which coverage schemes they "
            f"execute well and which they struggle in. Use DEPM and the defensive stat "
            f"profile to contextualize how impactful they are relative to the rest of the "
            f"league — note any area where they are a clear outlier, positive or "
            f"negative.\n\n"
            f"Paragraph 2 — OFFENSIVE PROFILES THAT EXPLOIT THEM: Use the neighborhood "
            f"data to identify which offensive archetypes consistently score on them and "
            f"the physical reason why that matchup works. Then identify which offensive "
            f"profiles they handle well and the mechanical reason their length, "
            f"positioning, or scheme disrupts those players. Do not rely on PPP alone — "
            f"also use field goal percentage, efficiency, and turnovers to fully "
            f"contextualize those results.\n\n"
            f"Paragraph 3 — TENDENCY AND TRAJECTORY: If career trajectory data shows a "
            f"trend that changes the attacking read — declining lateral quickness, "
            f"changing defensive role, foul-rate patterns — state it and explain the "
            f"implication. Only reference {player.name}'s offensive role if their scoring "
            f"load forces the opponent to make trade-offs in how they can load toward "
            f"their defensive assignment. Do not write a career section.\n\n"
            f"Paragraph 4 — THE VERDICT: One specific scheme recommendation naming the "
            f"play type, the action to run at them, and the floor zone where the offense "
            f"should initiate. Every number in this report supports an argument about how "
            f"to score against them.\n\n{context}"
        )


def generate_player_profile_report(
    player: Player,
    role: str,
    neighborhood: List[Dict],
    api_key: str,
    shot_zones: Optional[Dict] = None,
    career_df: Optional[pd.DataFrame] = None,
) -> str:
    """Generate a player profile report (blocking, returns full string)."""
    prompt = _build_profile_prompt(player, role, neighborhood, shot_zones, career_df)
    return _call_anthropic(prompt, api_key)


def stream_player_profile_report(
    player: Player,
    role: str,
    neighborhood: List[Dict],
    api_key: str,
    shot_zones: Optional[Dict] = None,
    career_df: Optional[pd.DataFrame] = None,
):
    """Streaming variant — yields text chunks as they arrive from the API."""
    prompt = _build_profile_prompt(player, role, neighborhood, shot_zones, career_df)
    return stream_report(prompt, api_key)


def _fmt_similar_scorers(similar_list: List[Dict], top_n: int = 5) -> str:
    if not similar_list:
        return "No similar scorers found."
    lines = ["Scorers with most similar offensive profile:"]
    for s in similar_list[:top_n]:
        lines.append(
            f"  • {s['scorer']} ({s.get('team','')}, {s.get('position','')}): "
            f"MPS_off={s['combined_score']:.2f}, shared defenders={s['shared_opponents']}"
        )
    return "\n".join(lines)


def generate_similarity_report(
    target: Player,
    similar_list: List[Dict],
    graph_obj: MatchupGraph,
    api_key: str,
    role: str = "auto",
) -> str:
    """
    Generate a similarity scouting report for either a scorer or a defender.

    role: "offense", "defense", or "auto" (detected from similar_list keys).
    """
    if role == "auto":
        role = "offense" if similar_list and "scorer" in similar_list[0] else "defense"

    if role == "offense":
        target_neighborhood = graph_obj.get_offensive_neighborhood(target.name, top_n=8)
        similar_fmt = _fmt_similar_scorers(similar_list, top_n=5)
        role_label = "SCORER"
        neighborhood_label = "OFFENSIVE MATCHUP PROFILE"
    else:
        target_neighborhood = graph_obj.get_defensive_neighborhood(target.name, top_n=8)
        similar_fmt = _fmt_similar_defenders(similar_list, top_n=5)
        role_label = "DEFENDER"
        neighborhood_label = "DEFENSIVE MATCHUP PROFILE"

    context = f"""
=== TARGET {role_label} ===
{_fmt_player_bio(target)}

=== {neighborhood_label} ===
{_fmt_neighborhood_summary(target_neighborhood, role=role, top_n=6)}

=== SIMILAR PLAYERS ===
{similar_fmt}
""".strip()

    n = min(5, len(similar_list))
    similar_names = ", ".join(s.get("defender", s.get("scorer", "?")) for s in similar_list[:n])

    prompt = (
        f"Write a scouting report on {target.name} and the {n} most similar players "
        f"to them, covering both offensive and defensive profiles. The {n} most "
        f"similar players by combined MPS score are: {similar_names}. "
        f"Reference these players by name throughout the report. Four paragraphs.\n\n"
        f"Paragraph 1 — PLAYER ARCHETYPE: What archetype does {target.name} represent "
        f"on both ends. What physical and schematic traits define how they score and "
        f"how they guard. Which floor zones they own offensively and which coverage "
        f"responsibilities they handle defensively. This is the baseline — everything "
        f"in the report connects back to it.\n\n"
        f"Paragraph 2 — OFFENSIVE SIMILARITY: What the comparable scorers share with "
        f"{target.name} mechanically — not just that the numbers align, but why they "
        f"attack the same zones, exploit the same defensive archetypes, and struggle "
        f"against the same coverage types. Use the MPS_off score and shared defender "
        f"data to explain which parts of their offensive games are truly "
        f"interchangeable and which are only superficially similar. Name specific "
        f"players from the list when the comparison is mechanically meaningful. Name "
        f"the offensive archetype this group represents.\n\n"
        f"Paragraph 3 — DEFENSIVE SIMILARITY: What the comparable defenders share "
        f"with {target.name} mechanically — why the same offensive archetypes give "
        f"them all trouble and why the same offensive profiles get contained. Use the "
        f"MPS_def score and shared opponent data to explain which parts of their "
        f"defensive profiles are truly interchangeable. Name specific players from "
        f"the list when the comparison is mechanically meaningful. Name the defensive "
        f"archetype this group represents and what that means for how offenses should "
        f"attack any player in it.\n\n"
        f"Paragraph 4 — THE VERDICT: One actionable conclusion for a front office or "
        f"coaching staff. If {target.name} is unavailable, name the single most viable "
        f"replacement from the {n} similar players on each end and the specific reason "
        f"why — not just that the scores are close, but what they share mechanically. "
        f"If the context is game-planning against {target.name}, state which tendency "
        f"from the comparable players group best predicts their coverage or attack "
        f"pattern. Commit to one conclusion — roster move, scheme adjustment, or "
        f"defensive assignment.\n\n{context}"
    )
    return _call_anthropic(prompt, api_key)



# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _sanitize(text: str) -> str:
    """Normalize typographic characters; leave all Unicode intact (Anthropic API is UTF-8)."""
    return (
        text
        .replace("\u2019", "'").replace("\u2018", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2013", "-").replace("\u2014", "-")
        .replace("\u2026", "...")
    )


def _call_anthropic(user_prompt: str, api_key: str, max_tokens: int = 2048) -> str:
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": _sanitize(SYSTEM_PROMPT),
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[
                {"role": "user", "content": _sanitize(user_prompt)},
            ],
        )
        return message.content[0].text
    except anthropic.AuthenticationError:
        return "ERROR:Invalid Anthropic API key. Please check your key in the sidebar."
    except anthropic.RateLimitError:
        return "ERROR:Anthropic rate limit hit. Please wait a moment and try again."
    except Exception as e:
        return f"ERROR:Report generation failed: {e}"


def stream_report(user_prompt: str, api_key: str):
    """
    Yield text chunks from a streaming Anthropic response.
    Uses ephemeral prompt caching on the system prompt — first call pays full
    price; subsequent calls within 5 minutes are served from cache at lower
    latency and cost.
    Yields a single error string on failure (so callers always get a string).
    """
    try:
        client = anthropic.Anthropic(api_key=api_key)
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=[{
                "type": "text",
                "text": _sanitize(SYSTEM_PROMPT),
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": _sanitize(user_prompt)}],
        ) as stream:
            for text in stream.text_stream:
                yield text
    except anthropic.AuthenticationError:
        yield "ERROR:Invalid Anthropic API key. Please check your key in the sidebar."
    except anthropic.RateLimitError:
        yield "ERROR:Anthropic rate limit hit. Please wait a moment and try again."
    except Exception as e:
        yield f"ERROR:Report generation failed: {e}"