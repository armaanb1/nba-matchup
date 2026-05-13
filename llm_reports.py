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
    worst = sorted(valid, key=lambda x: x[ppp_key], reverse=(role != "offense"))[:top_n]

    def _fmt_row(r):
        team = r.get(team_key) or ""
        team_str = f" ({team})" if team else ""
        return f"  • vs {r[opp_key]}{team_str}: PPP {r[ppp_key]:.3f}  ({r['possessions']:.0f} poss)"

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


def fmt_career_context(career_df: pd.DataFrame, player_name: str) -> str:
    """Public alias — formats career splits for use as LLM context."""
    return _fmt_career_trajectory(career_df, player_name)


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
    "You are a veteran NBA scout writing an internal report for a coaching staff preparing for a playoff series. "
    "You have watched every game. You are not summarizing a spreadsheet — you are telling the coaches something "
    "they need to know that they might not see just by looking at the numbers themselves. "

    "Voice and tone rules — these are non-negotiable: "

    "1. Never lead a sentence with a stat, but every analytical claim must be anchored by one. "
    "Stats belong parenthetically inside the argument, not as the opening. "
    "Wrong: '46.7% FG and 0.298 PPP this season.' "
    "Right: 'Brunson lives in the mid-range and the paint — and he's efficient there (46.7% FG, 0.298 PPP) "
    "because his frame actually works in his favor in those zones, not against him.' "
    "If you make a claim about tendency, decline, improvement, or comparison — cite the number. "
    "Only use numbers present in the data provided to you. Never invent or approximate a statistic. "

    "2. Every cited number must be followed by a mechanical explanation. What physical attribute, skill, or "
    "defensive scheme produced that number? If you cannot explain why a number exists, do not cite it. "

    "3. Write in complete thoughts, not bullet fragments. Each paragraph should make one argument, "
    "support it with evidence, and connect it to a tactical consequence. "

    "4. Use the language scouts actually use. Paint touches. Leverage points. Help rotations. "
    "Verticality. Shade coverage. Screen navigation. Drop vs. hedge vs. switch. "
    "Do not use phrases like 'it is worth noting', 'demonstrates', 'exhibits', or 'showcases'. "

    "5. Archetypes over individuals. When describing who a player beats or struggles against, "
    "lead with the archetype — what physical or schematic type causes the outcome — then name "
    "the specific players as examples. "

    "6. The head-to-head matchup section must be the sharpest part of the report. "
    "This is what the coach actually needs. Open with the headline finding from the data — "
    "the single most important thing the PPP, FG%, or possession count reveals about this pairing. "
    "Then give the physical reason it happened. Then state the game-plan implication. "
    "Do not set up the matchup with background; get to the finding immediately. "

    "7. The strategic recommendation must be a specific action with a specific trigger. "
    "It must reference the offensive player's weakest high-volume zone and the defensive player's "
    "best physical tool. No generic advice. "

    "8. When career trajectory data is provided, do not write a career section. "
    "Reference the trend only where it changes the tactical read — for example, if a player's "
    "3P% has declined three straight seasons, mention it when discussing how hard to chase them "
    "off the line. If the trend confirms what this season already shows, skip it. "
    "Current season stats always lead. "

    "9. No title, header, bullet points, or preamble anywhere in the report — not even section labels. "
    "Write continuous prose. Start directly with the offensive profile. "
    "Structure your four paragraphs as follows: "
    "(1) OFFENSIVE PROFILE — where the player operates, why those zones work given their physical tools, "
    "which shot zones are weapons vs. volume releases; "
    "(2) DEFENSIVE ARCHETYPE ANALYSIS — which archetypes contain them and why mechanically, "
    "which archetypes struggle and the physical reason; "
    "(3) THE SPECIFIC MATCHUP — what the head-to-head data shows about this exact pairing, "
    "the physical explanation, and the game-plan consequence; "
    "(4) STRATEGIC RECOMMENDATION — one specific triggered action naming play type, coverage scheme, "
    "and the floor zone to force the offensive player toward. "
    "Total report: 400-500 words. "

    "10. Write about the offensive player first, in full, before addressing the defensive player. "
    "The offensive profile must describe: where on the floor they operate, why they are effective "
    "there given their physical tools, which shot zones are genuine weapons vs. volume releases, "
    "and which defensive archetypes give them trouble and why. Only then move to the defender."
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

    if off_zone_ctx or def_zone_ctx:
        zone_profile_instruction = (
            f" Shot distribution data is provided — use it inside the offensive profile to specify "
            f"exactly where {off_player.name} attacks, distinguish genuine weapons from volume zones, "
            f"and contrast corner 3 vs. above-break 3 tendencies and paint frequency. "
            f"Do not list the zones — draw conclusions about how the defense must position."
        )
    else:
        zone_profile_instruction = ""

    defender_offense_instruction = (
        f" Only reference {def_player.name}'s own offensive role if his usage or scoring load is "
        f"notably high for a player in this defensive assignment — and only if it changes how the "
        f"coaching staff should manage his minutes or deployment."
    )

    return (
        f"Write a scouting report on {off_player.name} as the offensive player being guarded by {def_player.name}. "
        f"The report is for a coaching staff preparing a defensive game plan. "
        f"Start with {off_player.name}'s offensive profile — where he operates, why those zones work given his physical tools, "
        f"which archetypes struggle to contain him and why, and which archetypes neutralize him and the mechanical reason.{zone_profile_instruction} "
        f"Then analyze {def_player.name} specifically as the matchup — what his physical profile means for this pairing "
        f"and what the head-to-head data shows about whether that expectation held.{defender_offense_instruction} "
        f"Close with one specific, triggered strategic recommendation that names the play type, "
        f"the coverage scheme, and the floor zone to force {off_player.name} toward. "
        f"Use the stats in the context below as evidence — weave them parenthetically into the argument.\n\n{context}"
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
        if zone_ctx:
            zone_instruction = (
                f" Shot distribution data is provided — use it to identify exactly where {player.name} "
                f"operates, which zones are genuine weapons vs. pressure releases, and what that means "
                f"for how a defense must load and position. Do not list the zones. Draw conclusions from them."
            )
        else:
            zone_instruction = ""

        return (
            f"Write a scouting report on {player.name} as an offensive player, for a coaching staff "
            f"preparing to defend him. Answer: where does he score, why does it work given his physical tools, "
            f"and what schematic or physical conditions shut him down.{zone_instruction} "
            f"The neighborhood data shows his PPP against each defender he has faced — use it to identify "
            f"the defensive archetype that limits him and explain the physical reason why that body type "
            f"disrupts his game, then name the archetype that gets exploited and why. "
            f"If career trajectory data shows a trend that changes the defensive read — declining FTA, "
            f"three-point volume without efficiency, slipping rim conversion — weave it in. Do not write a career section. "
            f"Close with one specific scheme recommendation naming the play type, the coverage scheme, "
            f"and the floor zone to force {player.name} toward. "
            f"Every number supports an argument about how to guard him.\n\n{context}"
        )
    else:
        return (
            f"Write a scouting report on {player.name} as a defender, for a coaching staff preparing "
            f"to attack him. Answer: how does he guard, what are his physical tools and limitations, "
            f"and which offensive profiles consistently score on him. "
            f"The neighborhood data shows PPP allowed per offensive player he has guarded — use it to identify "
            f"the offensive archetype that exploits him and explain the physical reason why that matchup "
            f"works, then identify what archetype he handles well and why his length, positioning, or scheme "
            f"disrupts those players. "
            f"His own offensive profile is relevant only if his scoring load forces the opponent to make "
            f"trade-offs — e.g. if he is a threat that commands attention off-ball and that changes how "
            f"a defense can load toward his assignment. "
            f"If career trajectory data shows a trend that changes the attacking read — declining lateral "
            f"quickness, changing role, foul-rate patterns — weave it in. Do not write a career section. "
            f"Close with one specific scheme recommendation naming the play type, the action to run at him, "
            f"and the floor zone where the offense should initiate. "
            f"Every number supports an argument about how to score against him.\n\n{context}"
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


def generate_similarity_report(
    target: Player,
    similar_list: List[Dict],
    graph_obj: MatchupGraph,
    api_key: str,
) -> str:
    """Generate a report explaining defensive similarity and strategic implications."""
    target_neighborhood = graph_obj.get_defensive_neighborhood(target.name, top_n=8)

    context = f"""
=== TARGET DEFENDER ===
{_fmt_player_bio(target)}

=== DEFENSIVE MATCHUP PROFILE ===
{_fmt_neighborhood_summary(target_neighborhood, role='defense', top_n=6)}

=== SIMILAR DEFENDERS ===
{_fmt_similar_defenders(similar_list, top_n=5)}
""".strip()

    prompt = (
        f"Write a scouting report explaining what defenders are most similar to "
        f"{target.name} and what this means strategically. "
        f"Lead with the defensive archetype {target.name} represents — what physical and schematic "
        f"traits define how they guard, and which offensive profiles exploit or struggle against that archetype. "
        f"Then explain what the similar defenders share with {target.name} mechanically — "
        f"not just that the numbers match, but why the same offensive players give them all trouble. "
        f"Close with what a front office or coaching staff should know when game-planning against "
        f"any defender in this archetype group.\n\n{context}"
    )
    return _call_anthropic(prompt, api_key)


def generate_team_matchup_report(
    team1_name: str,
    team2_name: str,
    team1_stats: Dict,
    team2_stats: Dict,
    graph_obj: MatchupGraph,
    api_key: str,
) -> str:
    """Generate a scouting report for a team vs. team matchup."""
    context = f"""
=== TEAM 1: {team1_name.upper()} ===
{_fmt_team_stats(team1_name, team1_stats)}

=== TEAM 2: {team2_name.upper()} ===
{_fmt_team_stats(team2_name, team2_stats)}

=== KEY PLAYER MATCHUPS (from matchup graph) ===
{_fmt_cross_team_matchups(team1_name, team2_name, graph_obj)}
""".strip()

    prompt = (
        f"Write a team matchup scouting report for {team1_name} vs {team2_name}. "
        f"For each advantage cited, explain the schematic or physical reason behind it — not just the number. "
        f"Identify the two or three individual matchups that will determine the series outcome and explain "
        f"why those specific pairings matter physically and schematically. "
        f"Each strategic recommendation must name the play type, screening action, or coverage scheme — "
        f"not 'exploit the mismatch'. "
        f"Do not list stats. Build an argument about which team wins and why.\n\n{context}"
    )
    return _call_anthropic(prompt, api_key)


def generate_playoff_matchup_keys(
    team1_name: str,
    team2_name: str,
    team1_stats: Dict,
    team2_stats: Dict,
    seed1: int,
    seed2: int,
    series_prob: float,
    graph_obj: MatchupGraph,
    api_key: str,
    roster_t1: Optional[List[str]] = None,
    roster_t2: Optional[List[str]] = None,
) -> str:
    """Generate LLM-powered 'keys to the series' for a playoff matchup."""
    _roster_section = ""
    if roster_t1 or roster_t2:
        _roster_section = "\n=== CURRENT ROSTERS (only reference these players) ===\n"
        if roster_t1:
            _roster_section += f"{team1_name}: {', '.join(roster_t1)}\n"
        if roster_t2:
            _roster_section += f"{team2_name}: {', '.join(roster_t2)}\n"

    context = f"""
=== PLAYOFF MATCHUP ===
#{seed1} {team1_name} vs #{seed2} {team2_name}
Projected series win probability: {team1_name} {series_prob:.0%} / {team2_name} {1-series_prob:.0%}

=== {team1_name.upper()} TEAM STATS ===
{_fmt_team_stats(team1_name, team1_stats)}

=== {team2_name.upper()} TEAM STATS ===
{_fmt_team_stats(team2_name, team2_stats)}
{_roster_section}
=== KEY PLAYER MATCHUPS (from season matchup graph) ===
{_fmt_cross_team_matchups(team1_name, team2_name, graph_obj)}
""".strip()

    roster_instruction = (
        " Only reference players who appear in the provided rosters — do not mention any player "
        "not listed there."
        if (roster_t1 or roster_t2) else ""
    )

    prompt = (
        f"Write a 'Keys to the Series' breakdown for #{seed1} {team1_name} vs #{seed2} {team2_name}. "
        f"Identify 3-4 keys. For each one: state the specific tactical decision that must be made, "
        f"explain WHY it matters physically or schematically — not just what the number says — "
        f"and give a concrete scheme action to address it. "
        f"Name the play type, the coverage scheme, or the rotation — not 'exploit the mismatch'. "
        f"Do not list stats. Make an argument about what decides this series.{roster_instruction} "
        f"Keep it under 350 words.\n\n{context}"
    )
    return _call_anthropic(prompt, api_key)


def _fmt_team_stats(team_name: str, stats: Dict) -> str:
    """Format team stats dict into a readable string for LLM context."""
    lines = [f"Team: {team_name}"]
    stat_map = [
        ("Off Rating",   "OFF_RATING",   ".1f"),
        ("Def Rating",   "DEF_RATING",   ".1f"),
        ("Net Rating",   "NET_RATING",   ".1f"),
        ("Pace",         "PACE",         ".1f"),
        ("eFG%",         "EFG_PCT",      ".1%"),
        ("TOV%",         "TM_TOV_PCT",   ".1%"),
        ("OReb%",        "OREB_PCT",     ".1%"),
        ("TS%",          "TS_PCT",       ".1%"),
        ("PPG",          "PTS",          ".1f"),
        ("Opp PPG",      "OPP_PTS",      ".1f"),
        ("W-L",          "W",            None),
    ]
    parts = []
    for label, key, fmt in stat_map:
        val = stats.get(key)
        if val is None:
            continue
        if fmt is None:
            losses = stats.get("L")
            if losses is not None:
                parts.append(f"Record: {int(val)}-{int(losses)}")
        else:
            try:
                parts.append(f"{label}: {float(val):{fmt}}")
            except (TypeError, ValueError):
                pass
    lines.extend(parts)
    return "\n".join(lines)


def _fmt_cross_team_matchups(team1_name: str, team2_name: str, graph_obj: MatchupGraph, top_n: int = 6) -> str:
    """Find player matchup edges between players from team1 (offense) vs team2 (defense)
    and vice versa, formatted for LLM context."""
    lines = []
    matchups_found = []

    for (off_id, def_id), edge in graph_obj.matchups.items():
        off_p = graph_obj.players.get(off_id)
        def_p = graph_obj.players.get(def_id)
        if not off_p or not def_p:
            continue
        off_team = (off_p.team or "").lower()
        def_team = (def_p.team or "").lower()
        t1l = team1_name.lower()
        t2l = team2_name.lower()

        t1_words = set(t1l.split())
        t2_words = set(t2l.split())
        off_match_t1 = any(w in off_team for w in t1_words if len(w) > 3)
        off_match_t2 = any(w in off_team for w in t2_words if len(w) > 3)
        def_match_t1 = any(w in def_team for w in t1_words if len(w) > 3)
        def_match_t2 = any(w in def_team for w in t2_words if len(w) > 3)

        if (off_match_t1 and def_match_t2) or (off_match_t2 and def_match_t1):
            matchups_found.append((edge.points_per_possession, off_p.name, def_p.name, edge))

    if not matchups_found:
        return "No direct cross-team matchup data found in the graph for this season."

    matchups_found.sort(key=lambda x: x[3].possessions, reverse=True)

    lines.append(f"Top {min(top_n, len(matchups_found))} cross-team matchups by possessions:")
    for ppp, off_name, def_name, edge in matchups_found[:top_n]:
        lines.append(
            f"  • {off_name} (off) vs {def_name} (def): "
            f"PPP {edge.points_per_possession:.3f}, {edge.possessions:.0f} poss, "
            f"FG {edge.fg_pct:.1%}, {edge.games_played}g"
        )
    return "\n".join(lines)


def generate_game_prep_report(
    off_players: List[Player],
    def_players: List[Player],
    graph_obj: MatchupGraph,
    api_key: str,
) -> str:
    """Generate a pre-game matchup preparation report."""
    sections = []
    for off in off_players[:4]:
        hood = graph_obj.get_offensive_neighborhood(off.name, top_n=5)
        sections.append(f"--- {off.name} (offense) ---")
        sections.append(_fmt_player_bio(off))
        sections.append(_fmt_neighborhood_summary(hood, role="offense", top_n=4))

    context = "\n\n".join(sections)
    prompt = (
        f"Write a pre-game scouting report for the upcoming matchup. "
        f"For each offensive player listed, identify which defenders give them trouble "
        f"and which they can exploit — and explain the physical or schematic reason why in each case. "
        f"Close with specific scheme recommendations for the defensive game plan.\n\n" + context
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


ANALYST_SYSTEM_PROMPT = (
    "You are a veteran NBA analyst and scout with 20 years of experience working in front offices and on coaching staffs. "
    "You think in terms of schemes, archetypes, and physical matchups — not just statistics. "
    "When someone asks you a basketball question, you answer it the way a scout would brief a coaching staff: "
    "direct, specific, grounded in how the game is actually played. "

    "Rules: "
    "1. Answer the question that was actually asked. Do not reframe it, hedge it, or answer a safer version of it. "
    "2. Lead with the core answer, not background context. The person asking already knows basketball. "
    "3. Use scout language: scheme names, coverage types, play actions, positional archetypes, physical tools. "
    "Drop coverage. ICE. Switching. Spain pick-and-roll. DHO. Nail help. Weak-side skip. "
    "4. When a concept has nuance, explain the nuance — then commit to the most defensible position "
    "and explain your reasoning. Do not present both sides equally and leave the person to decide. "
    "5. Cite real players or real teams as examples where they make the argument sharper. "
    "6. Anchor every analytical claim with a specific statistic in parentheses. "
    "Narrative without numbers is opinion, not analysis. "
    "Simple decision rule for every stat: "
    "STEP 1 — Scan the VERIFIED CAREER DATA at the top of the prompt. "
    "If the number is there, cite the exact figure. Never say a stat is unavailable if it appears in that table. "
    "STEP 2 — If the number genuinely is not in the data, state the claim qualitatively — no estimates, no approximations. "
    "STEP 3 — Never be vague ('mid-30s range', 'roughly X', 'around Y') when the exact figure is in the data. "
    "Vague numbers are worse than no numbers. "
    "The data section is the only permitted source of specific statistics. "
    "7. All player teams, head coaches, and roster facts must come from the injected data only — "
    "never from training knowledge, which may be a year or more out of date. "
    "If a player's team or a team's head coach is not listed in the data, do not name one. "
    "8. Write in complete paragraphs. No bullet points. No numbered lists. "
    "No hedging phrases like 'it depends', 'great question', 'certainly', or 'as an AI.' "
    "9. Target 250-400 words. Long enough to be substantive, short enough to be useful in a film session."
)


def _call_anthropic(user_prompt: str, api_key: str, system_override: Optional[str] = None) -> str:
    system = system_override if system_override else SYSTEM_PROMPT
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=[{
                "type": "text",
                "text": _sanitize(system),
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
            max_tokens=1024,
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