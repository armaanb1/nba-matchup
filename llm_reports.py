"""
LLM Scouting Report generation using the Anthropic API.

The report synthesizes matchup graph data — edge weights, neighborhood context,
similar-player comparisons — into a natural-language scouting narrative.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import anthropic

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
    label = "best offensive matchups (highest PPP scored)" if role == "offense" \
        else "best defensive matchups (lowest PPP allowed)"

    best = sorted(rows, key=lambda x: x[ppp_key], reverse=(role == "offense"))[:top_n]
    worst = sorted(rows, key=lambda x: x[ppp_key], reverse=(role != "offense"))[:top_n]

    lines = [f"Top {top_n} {label}:"]
    for r in best:
        lines.append(f"  • vs {r[opp_key]}: PPP {r[ppp_key]:.3f}  ({r['possessions']:.0f} poss)")
    lines.append(f"\nBottom {top_n} (toughest matchups):")
    for r in worst:
        lines.append(f"  • vs {r[opp_key]}: PPP {r[ppp_key]:.3f}  ({r['possessions']:.0f} poss)")
    return "\n".join(lines)


def _fmt_shot_zones(zone_summary: Dict, player_name: str) -> str:
    """Format shot zone summary dict into LLM-readable context string."""
    if not zone_summary:
        return ""

    # Sort zones by frequency descending
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

    # Highlight corner 3 vs above-break 3 if both present
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
    "You are an NBA front office analyst writing an internal scouting document. "
    "Your audience is coaching staff who need to make tactical decisions, not fans who want to read about stats. "
    "Rules: "
    "1. Never state a PPP or efficiency number without immediately explaining WHY the player produced that number. "
    "Connect it to a physical attribute (height, wingspan, lateral speed), a skill (release point, handle, passing vision), "
    "or a schematic factor (coverage type, screening action, transition vs. halfcourt). "
    "2. Group matchup analysis by defensive archetype, not individual names. "
    "Example: 'Markkanen torches guards and drop-coverage bigs because his 7-1 frame with a perimeter skill set creates "
    "an impossible closeout problem.' Then list the specific players who fit that archetype. "
    "3. When describing a player's struggles, identify the defensive archetype that causes problems. "
    "4. Every Strategic Recommendation must be a specific action, not a general concept. "
    "Bad: 'Exploit mismatches.' Good: 'Run Spain pick-and-rolls with Markkanen as the screener to force a guard switch.' "
    "5. Never reference players not in the provided roster data. "
    "6. Do not add any title, header, subtitle, or preamble to the report. "
    "Start directly with the first section heading. No 'INTERNAL SCOUTING REPORT', "
    "no 'Prepared for Coaching Staff', no matchup title line — none of that. "
    "Structure the report with short labeled sections. Aim for 300-450 words."
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
) -> str:
    """Generate a scouting report for a specific head-to-head matchup."""
    off_zone_ctx = _fmt_shot_zones(off_shot_zones or {}, off_player.name)
    def_zone_ctx = _fmt_shot_zones(def_shot_zones or {}, def_player.name)

    zone_section = ""
    if off_zone_ctx or def_zone_ctx:
        zone_section = "\n\n=== SHOT DISTRIBUTION ==="
        if off_zone_ctx:
            zone_section += f"\n{off_zone_ctx}"
        if def_zone_ctx:
            zone_section += f"\n\n{def_zone_ctx}"

    context = f"""
=== OFFENSIVE PLAYER ===
{_fmt_player_bio(off_player)}

=== DEFENSIVE PLAYER ===
{_fmt_player_bio(def_player)}

=== HEAD-TO-HEAD MATCHUP ===
{_fmt_matchup(edge, off_player.name, def_player.name)}

=== {off_player.name.upper()} OFFENSIVE CONTEXT ===
{_fmt_neighborhood_summary(off_neighborhood, role='offense')}

=== {def_player.name.upper()} DEFENSIVE CONTEXT ===
{_fmt_neighborhood_summary(def_neighborhood, role='defense')}{zone_section}
""".strip()

    shot_zone_instruction = (
        " Reference the shot distribution data to explain where on the floor the offense attacks "
        "and how the defense should position — specifically call out corner 3 vs. above-break 3 "
        "tendencies and paint frequency."
        if (off_zone_ctx or def_zone_ctx) else ""
    )

    prompt = (
        f"Generate a scouting report for the matchup between "
        f"{off_player.name} (offense) and {def_player.name} (defense). "
        f"For each efficiency number cited, explain the physical or schematic reason behind it. "
        f"Identify the defensive archetype that either neutralizes or struggles against {off_player.name}."
        f"{shot_zone_instruction} "
        f"End with one concrete, specific strategic recommendation for the defending team.\n\n{context}"
    )
    return _call_anthropic(prompt, api_key)


def generate_player_profile_report(
    player: Player,
    role: str,
    neighborhood: List[Dict],
    api_key: str,
    shot_zones: Optional[Dict] = None,
) -> str:
    """Generate a player scouting report based on their full matchup profile."""
    zone_ctx = _fmt_shot_zones(shot_zones or {}, player.name)
    zone_section = f"\n\n=== SHOT DISTRIBUTION ===\n{zone_ctx}" if zone_ctx else ""

    context = f"""
=== PLAYER PROFILE ===
{_fmt_player_bio(player)}

=== MATCHUP NEIGHBORHOOD ({role.upper()}) ===
{_fmt_neighborhood_summary(neighborhood, role=role, top_n=6)}{zone_section}
""".strip()

    shot_zone_instruction = (
        " Use the shot distribution data to specify exactly where on the floor the player attacks "
        "or is attacked — corner 3 vs. above-break 3 tendencies, paint frequency, mid-range reliance."
        if zone_ctx else ""
    )

    prompt = (
        f"Generate a scouting report for {player.name} focusing on their "
        f"{'offensive' if role == 'offense' else 'defensive'} matchup profile. "
        f"For each pattern identified, explain WHY it happens — connect it to their physical profile, "
        f"skill set, or the schematic contexts where they thrive or struggle."
        f"{shot_zone_instruction} "
        f"Identify the defensive archetype that gives them the most trouble and explain the mechanical reason. "
        f"End with one specific, actionable scheme recommendation.\n\n{context}"
    )
    return _call_anthropic(prompt, api_key)


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
        f"Generate a scouting report explaining what defenders are most similar to "
        f"{target.name} and what this means strategically. "
        f"Discuss the types of scorers they share difficulty/success against, "
        f"and what front offices or coaches could learn from this similarity analysis. "
        f"Cite the data.\n\n{context}"
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
    """Generate a scouting report for a team vs. team matchup, incorporating
    player-level matchup data from the graph and team-level stats."""
    context = f"""
=== TEAM 1: {team1_name.upper()} ===
{_fmt_team_stats(team1_name, team1_stats)}

=== TEAM 2: {team2_name.upper()} ===
{_fmt_team_stats(team2_name, team2_stats)}

=== KEY PLAYER MATCHUPS (from matchup graph) ===
{_fmt_cross_team_matchups(team1_name, team2_name, graph_obj)}
""".strip()

    prompt = (
        f"Generate a comprehensive team matchup scouting report for {team1_name} vs {team2_name}. "
        f"For each advantage cited, explain the schematic or physical reason behind it — not just the number. "
        f"Group the individual matchup analysis by defensive archetype: which archetypes from each roster "
        f"create exploitation opportunities, and which neutralize threats. "
        f"Each strategic recommendation must be a specific scheme action (e.g., 'Run weak-side DHO actions "
        f"to force switches onto the center'). Cite specific numbers.\n\n{context}"
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
    """Generate LLM-powered 'keys to the series' for a playoff matchup.
    If live rosters are provided, they are included in the prompt so Claude
    only references players actually on each team.
    """
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
        f"Generate a 'Keys to the Series' breakdown for #{seed1} {team1_name} vs #{seed2} {team2_name}. "
        f"Identify 3-4 specific keys. For each key: (1) state the tactical decision that must be made, "
        f"(2) cite the specific stat or matchup data that makes it pivotal, "
        f"(3) explain WHY it matters physically or schematically — not just what the number says. "
        f"Each strategic recommendation must be a specific action: name the play type, "
        f"the screening action, or the coverage scheme — not 'exploit the mismatch'.{roster_instruction} "
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

    # Look through all edges for cross-team matchups
    for (off_id, def_id), edge in graph_obj.matchups.items():
        off_p = graph_obj.players.get(off_id)
        def_p = graph_obj.players.get(def_id)
        if not off_p or not def_p:
            continue
        off_team = (off_p.team or "").lower()
        def_team = (def_p.team or "").lower()
        t1l = team1_name.lower()
        t2l = team2_name.lower()

        # Match if off is from t1 and def is from t2, or vice versa
        # Use partial matching since team names may not be exact
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

    # Sort by possessions for the most significant matchups
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
        "Generate a pre-game scouting report for the upcoming matchup. "
        "For each offensive player listed, identify which defenders give them trouble "
        "and which ones they can exploit, based on the data. "
        "Include strategic recommendations.\n\n" + context
    )
    return _call_anthropic(prompt, api_key)


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _sanitize(text: str) -> str:
    """Replace common non-ASCII characters with ASCII equivalents."""
    return (
        text
        .replace("\u2019", "'").replace("\u2018", "'")   # curly single quotes
        .replace("\u201c", '"').replace("\u201d", '"')   # curly double quotes
        .replace("\u2013", "-").replace("\u2014", "-")   # en/em dash
        .replace("\u2026", "...")                         # ellipsis
        .encode("ascii", errors="replace").decode("ascii")
    )


def _call_anthropic(user_prompt: str, api_key: str) -> str:
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=_sanitize(SYSTEM_PROMPT),
            messages=[
                {"role": "user", "content": _sanitize(user_prompt)},
            ],
        )
        return message.content[0].text
    except anthropic.AuthenticationError:
        return "❌ Invalid Anthropic API key. Please check your key in the sidebar."
    except anthropic.RateLimitError:
        return "❌ Anthropic rate limit hit. Please wait a moment and try again."
    except Exception as e:
        return f"❌ Report generation failed: {e}"
