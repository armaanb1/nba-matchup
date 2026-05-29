"""
Core data models for the NBA Matchup Network.

Classes:
    Player         — graph node representing an NBA player
    MatchupEdge    — weighted edge between offensive and defensive player
    MatchupGraph   — bipartite NetworkX graph with all four interaction modes
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Archetype enums
# ---------------------------------------------------------------------------

class OffensiveArchetype(str, Enum):
    SHOT_CREATOR         = "Shot Creator"
    PRIMARY_BH           = "Primary Ball Handler"
    SECONDARY_BH         = "Secondary Ball Handler"
    SLASHER              = "Slasher"
    OFF_SCREEN_SHOOTER   = "Off Screen Shooter"
    MOVEMENT_SHOOTER     = "Movement Shooter"
    STATIONARY_SHOOTER   = "Stationary Shooter"
    ATHLETIC_FINISHER    = "Athletic Finisher"
    VERSATILE_BIG        = "Versatile Big"
    STRETCH_BIG          = "Stretch Big"
    POST_SCORER          = "Post Scorer"
    ROLL_AND_CUT_BIG     = "Roll & Cut Big"


class DefensiveRole(str, Enum):
    POINT_OF_ATTACK  = "Point of Attack"
    WING_STOPPER     = "Wing Stopper"
    CHASER           = "Chaser"
    HELPER           = "Helper"
    ANCHOR_BIG       = "Anchor Big"
    MOBILE_BIG       = "Mobile Big"
    LOW_ACTIVITY     = "Low Activity"


# ---------------------------------------------------------------------------
# Archetype classifier helpers
# ---------------------------------------------------------------------------

def _pctile(values: List[Optional[float]], pct: float) -> float:
    """Return the pct-th percentile of values, ignoring None."""
    arr = np.array([v for v in values if v is not None], dtype=float)
    return float(np.percentile(arr, pct)) if len(arr) > 0 else 0.0


def _is_big_by_position(position_str: Optional[str]) -> bool:
    """
    Proxy for center/PF when explicit position-time data is unavailable.
    Handles both full NBA API names ("Center", "Center-Forward") and
    abbreviations ("C", "F-C") since the API returns either depending on endpoint.
    """
    if not position_str:
        return False
    pos = position_str.strip()
    # Full names returned by CommonPlayerInfo
    if pos in ("Center", "Center-Forward", "Forward-Center"):
        return True
    # Abbreviations
    abbr = pos.upper().replace(" ", "").replace("-", "")
    return abbr in ("C", "FC", "CF")


def classify_scorer(
    stats: Dict,
    player_pool: List[Dict],
) -> Optional[OffensiveArchetype]:
    """
    Classify an offensive player's archetype from play-type frequencies and tracking.

    stats keys (all optional):
        scoring_possessions, pnr_bh_freq, iso_freq, post_freq, roll_freq, spot_freq,
        off_screen_freq, handoff_freq, cut_freq, putback_freq,
        drives_per75, fg3a_rate, paint_touches_per75, ast_per75,
        position_pct_c, position_pct_pf, position, avg_speed,
        catch_shoot_pct_of_3pa

    player_pool: list of stat dicts for all qualifying players (same keys).
    Returns None when critical play-type data is missing or sample < 250 poss.

    All percentile thresholds are computed within the current season's qualifying
    pool — no hardcoded absolute values.
    """
    scoring_poss = stats.get("scoring_possessions")
    if scoring_poss is not None and scoring_poss < 100:
        return None

    pnr_bh      = stats.get("pnr_bh_freq")
    iso         = stats.get("iso_freq")
    post        = stats.get("post_freq")
    roll        = stats.get("roll_freq")
    cut         = stats.get("cut_freq")
    putback     = stats.get("putback_freq")
    off_screen  = stats.get("off_screen_freq")
    handoff     = stats.get("handoff_freq")

    if all(v is None for v in [pnr_bh, iso, post, roll, cut, putback, off_screen, handoff]):
        return None  # no play-type data

    pnr_bh     = pnr_bh     or 0.0
    iso        = iso        or 0.0
    post       = post       or 0.0
    roll       = roll       or 0.0
    cut        = cut        or 0.0
    putback    = putback    or 0.0
    off_screen = off_screen or 0.0
    handoff    = handoff    or 0.0
    spot       = stats.get("spot_freq") or 0.0

    drives_per75    = stats.get("drives_per75")    or 0.0
    fg3a_rate       = stats.get("fg3a_rate")       or 0.0
    paint_touches   = stats.get("paint_touches_per75") or 0.0
    ast_per75       = stats.get("ast_per75")       or 0.0
    avg_speed       = stats.get("avg_speed")
    cs_pct          = stats.get("catch_shoot_pct_of_3pa")

    position_pct_c  = stats.get("position_pct_c")
    position_pct_pf = stats.get("position_pct_pf")
    position_str    = stats.get("position", "")

    qualifying = [
        p for p in player_pool
        if p.get("scoring_possessions") is None or (p.get("scoring_possessions") or 0) >= 100
    ]

    def _is_pool_big(p: Dict) -> bool:
        pct_c  = p.get("position_pct_c")
        pct_pf = p.get("position_pct_pf")
        if pct_c is not None or pct_pf is not None:
            pos_big = (pct_c or 0) >= 0.30 or (pct_pf or 0) >= 0.50
        else:
            pos_big = _is_big_by_position(p.get("position", ""))
        big_mix = (
            (p.get("roll_freq") or 0) + (p.get("cut_freq") or 0) +
            (p.get("putback_freq") or 0) + (p.get("post_freq") or 0)
        )
        return pos_big and big_mix >= 0.35

    big_pool = [p for p in qualifying if _is_pool_big(p)]
    gw_pool  = [p for p in qualifying if not _is_pool_big(p)]

    if position_pct_c is not None or position_pct_pf is not None:
        pct_c  = position_pct_c  or 0.0
        pct_pf = position_pct_pf or 0.0
        is_big_pos = pct_c >= 0.30 or pct_pf >= 0.50
    else:
        is_big_pos = _is_big_by_position(position_str)

    big_scoring_mix = roll + cut + putback + post
    is_big = is_big_pos and big_scoring_mix >= 0.35

    if is_big:
        active = sum([post >= 0.01, roll >= 0.01, cut >= 0.01, putback >= 0.01, spot >= 0.01])

        # Use Synergy spot_freq as the Stretch/Versatile Big gate — it directly
        # measures how often a player is used as a perimeter/catch-and-shoot threat.
        # fg3a_rate from EPM per-100 stats has scaling issues and produces false
        # positives (e.g., rim-runners classified as Stretch Bigs). spot_freq ≥ 12%
        # means at least 1 in 8 possessions is a spot-up play — a genuine floor role.
        _is_stretch = spot >= 0.12

        if _is_stretch and post >= 0.12 and active >= 3:
            return OffensiveArchetype.VERSATILE_BIG
        if _is_stretch and post < 0.12:
            return OffensiveArchetype.STRETCH_BIG
        if post >= 0.18:
            return OffensiveArchetype.POST_SCORER
        return OffensiveArchetype.ROLL_AND_CUT_BIG

    all_drives = [p.get("drives_per75") for p in qualifying]
    all_paint  = [p.get("paint_touches_per75") for p in qualifying]
    all_ast    = [p.get("ast_per75") for p in qualifying]
    gw_fg3a    = [p.get("fg3a_rate") for p in gw_pool]
    gw_drives  = [p.get("drives_per75") for p in gw_pool]
    all_speeds = [p.get("avg_speed") for p in gw_pool]
    all_cs     = [p.get("catch_shoot_pct_of_3pa") for p in gw_pool]

    p80_drives_lw   = _pctile(all_drives, 80)
    p50_drives_gw   = _pctile(gw_drives, 50)
    p75_initiator   = _pctile([v for v in (all_ast + all_paint) if v is not None], 75)
    p45_fg3a_gw     = _pctile(gw_fg3a, 45)
    p30_fg3a_gw     = _pctile(gw_fg3a, 30)
    p50_speed       = _pctile(all_speeds, 50)
    p50_cs          = _pctile(all_cs, 50)

    initiator_high = ast_per75 >= p75_initiator or paint_touches >= p75_initiator

    # Priority: Shot Creator > Primary BH > Slasher > Secondary BH >
    #           Athletic Finisher > Off Screen > Movement > Stationary
    if (pnr_bh + iso) >= 0.25 and iso >= 0.18 and drives_per75 >= p80_drives_lw:
        return OffensiveArchetype.SHOT_CREATOR

    if (pnr_bh + iso) >= 0.25 and iso < 0.18 and initiator_high:
        return OffensiveArchetype.PRIMARY_BH

    if drives_per75 >= p80_drives_lw and fg3a_rate < p30_fg3a_gw:
        return OffensiveArchetype.SLASHER

    if (pnr_bh + iso) >= 0.20 and iso < 0.18 and not initiator_high:
        return OffensiveArchetype.SECONDARY_BH

    if (cut + putback) >= 0.15 and drives_per75 < p50_drives_gw:
        return OffensiveArchetype.ATHLETIC_FINISHER

    if fg3a_rate >= p45_fg3a_gw and (off_screen + handoff) >= 0.12:
        return OffensiveArchetype.OFF_SCREEN_SHOOTER

    if fg3a_rate >= p45_fg3a_gw:
        if avg_speed is not None and avg_speed >= p50_speed:
            return OffensiveArchetype.MOVEMENT_SHOOTER
        if cs_pct is not None and cs_pct >= p50_cs:
            return OffensiveArchetype.STATIONARY_SHOOTER
        return OffensiveArchetype.MOVEMENT_SHOOTER  # default when movement data unavailable

    return None  # below all thresholds — not enough signal


def classify_defender(
    stats: Dict,
    player_pool: List[Dict],
) -> Optional[DefensiveRole]:
    """
    Classify a defensive player's role from matchup assignment and tracking data.

    stats keys (all optional):
        pct_time_vs_pg, pct_time_vs_sg, pct_time_vs_sf, pct_time_vs_pf, pct_time_vs_c,
        pct_time_vs_shot_creator, pct_time_vs_primary_bh, pct_time_vs_secondary_bh,
        pct_time_vs_slasher, pct_time_vs_off_screen, pct_time_vs_movement_shooter,
        pct_time_vs_stationary_shooter, pct_time_vs_athletic_finisher,
        pct_time_vs_versatile_big, pct_time_vs_post_scorer, pct_time_vs_stretch_big,
        pct_time_vs_roll_cut_big,
        matchup_difficulty,            — pre-computed z-score (see spec)
        def_positional_versatility,    — entropy of time at each position
        rim_time_pct,                  — restricted area FGA defended per 75
        off_ball_help_rate,            — deflections + off-ball steals + rim tags per 75
        height_inches, position

    Returns None when positional-assignment data is entirely missing.
    All percentile thresholds computed within the current season's qualifying pool.
    """
    vs_c   = stats.get("pct_time_vs_c")   or 0.0
    vs_pf  = stats.get("pct_time_vs_pf")  or 0.0
    vs_pg  = stats.get("pct_time_vs_pg")  or 0.0

    if all(stats.get(k) is None for k in (
        "pct_time_vs_pg", "pct_time_vs_sg", "pct_time_vs_sf",
        "pct_time_vs_pf", "pct_time_vs_c", "matchup_difficulty",
    )):
        return None

    rim_time       = stats.get("rim_time_pct")              or 0.0
    matchup_diff   = stats.get("matchup_difficulty")        or 0.0
    pos_versatility = stats.get("def_positional_versatility") or 0.0
    help_rate      = stats.get("off_ball_help_rate")        or 0.0
    height_in      = stats.get("height_inches")

    pool = player_pool or []
    rim_vals       = [p.get("rim_time_pct") for p in pool]
    diff_vals      = [p.get("matchup_difficulty") for p in pool]
    vers_vals      = [p.get("def_positional_versatility") for p in pool]
    help_vals      = [p.get("off_ball_help_rate") for p in pool]

    p80_rim    = _pctile(rim_vals, 80)
    p60_diff   = _pctile(diff_vals, 60)
    p50_diff   = _pctile(diff_vals, 50)
    p40_diff   = _pctile(diff_vals, 40)
    p50_vers   = _pctile(vers_vals, 50)
    p40_vers   = _pctile(vers_vals, 40)
    p60_help   = _pctile(help_vals, 60)

    # Big Gate: primarily guards bigs (≥55% time vs C/PF) OR high rim activity
    big_gate = (vs_c + vs_pf >= 0.55) or (rim_time >= p80_rim)

    if big_gate:
        # Anchor Big: concentrates on bigs, low positional versatility
        # (drops in PnR, stays home on roll/cut/post assignments)
        # Mobile Big: switches out to perimeter, guards mixed positions
        if pos_versatility < p40_vers:
            return DefensiveRole.ANCHOR_BIG
        return DefensiveRole.MOBILE_BIG

    vs_primary_bh   = stats.get("pct_time_vs_primary_bh")   or 0.0
    vs_shot_creator = stats.get("pct_time_vs_shot_creator")  or 0.0
    vs_slasher      = stats.get("pct_time_vs_slasher")       or 0.0
    vs_off_screen   = stats.get("pct_time_vs_off_screen")    or 0.0
    vs_movement     = stats.get("pct_time_vs_movement_shooter") or 0.0
    vs_stationary   = stats.get("pct_time_vs_stationary_shooter") or 0.0

    h = height_in or 78

    # Priority: Wing Stopper > Point of Attack > Chaser > Helper > Low Activity

    # Wing Stopper: tall enough to guard wings, guards tough shot creators,
    # and shows at least average positional versatility (can cover multiple spots).
    # Removed strict versatility gate — elite perimeter stoppers who
    # specialize in guarding one archetype should qualify via the archetype sum.
    if (
        (vs_shot_creator + vs_primary_bh + vs_slasher) >= 0.35
        and matchup_diff >= p60_diff
        and h >= 76
    ):
        return DefensiveRole.WING_STOPPER

    # Point of Attack: primarily guards ball-handlers; height ≤ 80" (6-8) to
    # distinguish from versatile wings. Using ≤ 80 rather than < 79 so that
    # 6-7 guards (79") are not excluded by a rounding artifact.
    if (
        (vs_pg + vs_primary_bh + vs_shot_creator) >= 0.40
        and matchup_diff >= p60_diff
        and h <= 80
    ):
        return DefensiveRole.POINT_OF_ATTACK

    if (
        (vs_off_screen + vs_movement + vs_stationary) >= 0.35
        and matchup_diff < p60_diff
    ):
        return DefensiveRole.CHASER

    if help_rate >= p60_help and matchup_diff < p50_diff:
        return DefensiveRole.HELPER

    if matchup_diff < p40_diff and pos_versatility < p40_vers and rim_time < _pctile(rim_vals, 40):
        return DefensiveRole.LOW_ACTIVITY

    return DefensiveRole.LOW_ACTIVITY  # default fallback


def classify_defensive_archetype(player: "Player") -> str:
    """Classify a player's defensive archetype from bio and per-100 stats."""
    h = player.height_inches or 0
    pos = (player.position or "").strip()
    # Normalize position to include abbrevs and full names
    pos_upper = pos.upper()
    has_C = "C" in pos_upper or "CENTER" in pos_upper
    has_F = "F" in pos_upper or "FORWARD" in pos_upper
    has_G = "G" in pos_upper or "GUARD" in pos_upper

    bpg      = player.bpg      or 0.0
    spg      = player.spg      or 0.0
    usg_pct  = player.usg_pct  or 0.0
    blk100   = player.p_blk_100 or 0.0
    stl100   = player.p_stl_100 or 0.0

    # 1. Anchor/Interior Big
    if has_C and h >= 81 and (bpg >= 1.2 or blk100 >= 3.0):
        return "Anchor/Interior Big"

    # 2. Mobile/Perimeter Big — 6-8+ (80") required; 6-7 is a wing, not a big
    if (has_C or has_F) and h >= 80 and not (has_C and h >= 81 and (bpg >= 1.2 or blk100 >= 3.0)):
        return "Mobile/Perimeter Big"

    # 3. Helper/Rotator — must be a non-guard big/wing; high-steal/block guards
    # (Derrick White, Ausar Thompson) are perimeter stoppers, not rotators.
    if h >= 77 and (bpg >= 0.8 or blk100 >= 2.0 or spg >= 1.5 or stl100 >= 2.5) and not has_G:
        return "Helper/Rotator"

    # 4. Wing Stopper — forwards/wings 6-4 to 6-9
    if 76 <= h <= 81 and (has_F or "SF" in pos_upper or "PF" in pos_upper or "SMALL FORWARD" in pos_upper or "POWER FORWARD" in pos_upper):
        return "Wing Stopper"

    # 5. Chaser — guards 6-2 to 6-7
    if 74 <= h <= 79 and (has_G or "SG" in pos_upper or "SF" in pos_upper or "SHOOTING GUARD" in pos_upper or "SMALL FORWARD" in pos_upper):
        return "Chaser"

    # 6. Low-Activity/Hider
    if usg_pct >= 0.28 and (player.spg is None or spg < 0.8) and (player.bpg is None or bpg < 0.4):
        return "Low-Activity/Hider"

    # 7. Point-of-Attack Defender
    if h <= 75 and has_G:
        return "Point-of-Attack Defender"

    # Default fallback by position
    if has_C:
        return "Anchor/Interior Big"
    if has_F:
        return "Wing Stopper"
    if has_G:
        return "Chaser"
    return "Wing Stopper"


def classify_offensive_archetype(player: "Player") -> str:
    """Classify a player's offensive archetype from bio and per-100 stats."""
    h = player.height_inches or 0
    pos = (player.position or "").strip()
    pos_upper = pos.upper()
    has_C = "C" in pos_upper or "CENTER" in pos_upper
    has_F = "F" in pos_upper or "FORWARD" in pos_upper

    apg      = player.apg      or 0.0
    usg_pct  = player.usg_pct  or 0.0
    fg3a_100 = player.p_fg3a_100    or 0.0
    rim_100  = player.p_fga_rim_100 or 0.0
    mid_100  = player.p_fga_mid_100 or 0.0

    # 1. Post Scorer
    if (has_C or "PF" in pos_upper or "POWER FORWARD" in pos_upper) and \
       player.p_fg3a_100 is not None and player.p_fg3a_100 < 2 and h >= 79:
        return "Post Scorer"

    # 2. Roll & Cut Big
    if (has_C or has_F) and rim_100 >= 6 and (player.p_fg3a_100 is None or fg3a_100 < 3):
        return "Roll & Cut Big"

    # 3. Primary Ball Handler
    if apg >= 5 or (apg >= 4 and usg_pct >= 0.25):
        return "Primary Ball Handler"

    # 4. Slasher
    if rim_100 >= 6 and (player.p_fg3a_100 is None or fg3a_100 < 4):
        return "Slasher"

    # 5. Spot-Up Shooter
    if fg3a_100 >= 6 and usg_pct < 0.22:
        return "Spot-Up Shooter"

    # 6. Shot Creator
    if usg_pct >= 0.25 and mid_100 >= 3:
        return "Shot Creator"

    # 7. Low-Usage Role Player
    if player.usg_pct is not None and usg_pct < 0.18:
        return "Low-Usage Role Player"

    # Default
    return "Versatile Scorer"


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

@dataclass
class Player:
    """NBA player node: bio + season stats + matchup-derived averages."""

    player_id: int
    name: str

    # Bio (from NBA API CommonPlayerInfo)
    position: Optional[str] = None
    team: Optional[str] = None
    height: Optional[str] = None          # e.g. "6-8"
    weight: Optional[int] = None          # lbs
    age: Optional[int] = None
    jersey: Optional[str] = None
    experience: Optional[int] = None      # seasons
    draft_year: Optional[str] = None
    draft_round: Optional[str] = None
    draft_pick: Optional[str] = None

    # Per-game stats (NBA API PlayerCareerStats / LeagueDashPlayerStats)
    ppg: Optional[float] = None
    rpg: Optional[float] = None
    apg: Optional[float] = None
    spg: Optional[float] = None
    bpg: Optional[float] = None
    tov: Optional[float] = None
    mpg: Optional[float] = None
    fg_pct: Optional[float] = None
    fg3_pct: Optional[float] = None
    ft_pct: Optional[float] = None
    ts_pct: Optional[float] = None
    games: Optional[int] = None

    # Advanced stats (NBA API LeagueDashPlayerStats Advanced)
    off_rating: Optional[float] = None
    def_rating: Optional[float] = None
    net_rating: Optional[float] = None
    usg_pct: Optional[float] = None
    pie: Optional[float] = None
    ast_pct: Optional[float] = None

    # EPM stats (dunksandthrees.com)
    epm_off: Optional[float] = None
    epm_def: Optional[float] = None
    epm_tot: Optional[float] = None

    # Per-100 possession stats (dunksandthrees.com)
    p_pts_100: Optional[float] = None
    p_ast_100: Optional[float] = None
    p_blk_100: Optional[float] = None
    p_stl_100: Optional[float] = None
    p_drb_100: Optional[float] = None
    p_orb_100: Optional[float] = None
    p_tov_100: Optional[float] = None
    p_fga_rim_100: Optional[float] = None
    p_fga_mid_100: Optional[float] = None
    p_fg3a_100: Optional[float] = None
    p_fgpct_rim: Optional[float] = None
    p_fgpct_mid: Optional[float] = None

    # Matchup-derived (computed from graph)
    avg_ppp_off: Optional[float] = None   # avg PPP scored on offense
    avg_ppp_def: Optional[float] = None   # avg PPP allowed on defense
    off_matchup_count: int = 0
    def_matchup_count: int = 0

    # Archetype labels (computed by classify_scorer / classify_defender)
    off_archetype: Optional[OffensiveArchetype] = None
    def_role: Optional[DefensiveRole] = None

    @property
    def height_inches(self) -> Optional[int]:
        """Convert height string '6-8' → 80 inches."""
        if not self.height:
            return None
        try:
            ft, inches = self.height.split("-")
            return int(ft) * 12 + int(inches)
        except (ValueError, AttributeError):
            return None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Player) and self.player_id == other.player_id

    def __hash__(self) -> int:
        return hash(self.player_id)

    def bio_dict(self) -> Dict:
        return {
            "Position": self.position or "—",
            "Team": self.team or "—",
            "Height": self.height or "—",
            "Weight": f"{self.weight} lbs" if self.weight else "—",
            "Age": str(self.age) if self.age else "—",
            "Experience": f"{self.experience} yrs" if self.experience else "—",
            "Jersey": f"#{self.jersey}" if self.jersey else "—",
            "Draft": (
                f"{self.draft_year} Rd {self.draft_round} Pick {self.draft_pick}"
                if self.draft_year and self.draft_year not in ("Undrafted", "")
                else (self.draft_year or "—")
            ),
        }

    def per_game_dict(self) -> Dict:
        def fmt_pct(v): return f"{v:.1%}" if v else "—"
        def fmt_f(v, d=1): return f"{v:.{d}f}" if v is not None else "—"
        return {
            "PPG": fmt_f(self.ppg),
            "RPG": fmt_f(self.rpg),
            "APG": fmt_f(self.apg),
            "SPG": fmt_f(self.spg),
            "BPG": fmt_f(self.bpg),
            "TOV": fmt_f(self.tov),
            "MPG": fmt_f(self.mpg),
            "FG%": fmt_pct(self.fg_pct),
            "3P%": fmt_pct(self.fg3_pct),
            "FT%": fmt_pct(self.ft_pct),
            "TS%": fmt_pct(self.ts_pct),
            "GP": str(self.games) if self.games else "—",
        }

    def advanced_dict(self) -> Dict:
        def fmt_f(v, d=1): return f"{v:.{d}f}" if v is not None else "—"
        def fmt_pct(v): return f"{v:.1%}" if v is not None else "—"
        return {
            "Off Rating": fmt_f(self.off_rating),
            "Def Rating": fmt_f(self.def_rating),
            "Net Rating": fmt_f(self.net_rating),
            "USG%": fmt_pct(self.usg_pct),
            "PIE": fmt_f(self.pie, d=3),
            "AST%": fmt_pct(self.ast_pct),
            "EPM": fmt_f(self.epm_tot, d=2),
            "OEPM": fmt_f(self.epm_off, d=2),
            "DEPM": fmt_f(self.epm_def, d=2),
            "PTS/100": fmt_f(self.p_pts_100, d=1),
            "AST/100": fmt_f(self.p_ast_100, d=1),
            "BLK/100": fmt_f(self.p_blk_100, d=1),
            "STL/100": fmt_f(self.p_stl_100, d=1),
            "DRB/100": fmt_f(self.p_drb_100, d=1),
            "ORB/100": fmt_f(self.p_orb_100, d=1),
            "TOV/100": fmt_f(self.p_tov_100, d=1),
            "Rim FGA/100": fmt_f(self.p_fga_rim_100, d=1),
            "Mid FGA/100": fmt_f(self.p_fga_mid_100, d=1),
            "3PA/100": fmt_f(self.p_fg3a_100, d=1),
            "Rim FG%": fmt_pct(self.p_fgpct_rim),
            "Mid FG%": fmt_pct(self.p_fgpct_mid),
        }


# ---------------------------------------------------------------------------
# MatchupEdge
# ---------------------------------------------------------------------------

class MatchupEdge:
    """Weighted edge: all stats for one offensive–defensive player pairing."""

    __slots__ = (
        "off_player_id", "def_player_id", "games_played", "possessions",
        "points", "fgm", "fga", "fg_pct", "fg3m", "fg3a", "fg3_pct",
        "assists", "turnovers", "blocks",
    )

    def __init__(
        self,
        off_player_id: int, def_player_id: int,
        games_played: int, possessions: float, points: float,
        fgm: float, fga: float, fg_pct: float,
        fg3m: float, fg3a: float, fg3_pct: float,
        assists: float, turnovers: float, blocks: float,
    ):
        self.off_player_id = off_player_id
        self.def_player_id = def_player_id
        self.games_played = games_played
        self.possessions = possessions
        self.points = points
        self.fgm = fgm
        self.fga = fga
        self.fg_pct = fg_pct
        self.fg3m = fg3m
        self.fg3a = fg3a
        self.fg3_pct = fg3_pct
        self.assists = assists
        self.turnovers = turnovers
        self.blocks = blocks

    @property
    def points_per_possession(self) -> float:
        return self.points / self.possessions if self.possessions > 0 else 0.0

    @property
    def effective_fg_pct(self) -> float:
        """eFG% = (FGM + 0.5 × FG3M) / FGA"""
        return (self.fgm + 0.5 * self.fg3m) / self.fga if self.fga > 0 else 0.0

    def to_dict(self) -> Dict:
        def fmt_pct(v): return f"{v:.1%}" if v else "—"
        return {
            "Possessions": f"{self.possessions:.1f}",
            "Points": f"{self.points:.1f}",
            "PPP": f"{self.points_per_possession:.3f}",
            "Games": self.games_played,
            "FGM-FGA": f"{self.fgm:.0f}-{self.fga:.0f}",
            "FG%": fmt_pct(self.fg_pct),
            "3PM-3PA": f"{self.fg3m:.0f}-{self.fg3a:.0f}" if self.fg3a else "—",
            "3P%": fmt_pct(self.fg3_pct) if self.fg3a else "—",
            "eFG%": fmt_pct(self.effective_fg_pct),
            "AST": f"{self.assists:.1f}",
            "TOV": f"{self.turnovers:.1f}",
            "BLK": f"{self.blocks:.1f}",
        }

    def __repr__(self) -> str:
        return (
            f"MatchupEdge(off={self.off_player_id}, def={self.def_player_id}, "
            f"PPP={self.points_per_possession:.2f}, Poss={self.possessions:.0f})"
        )


# ---------------------------------------------------------------------------
# MatchupGraph
# ---------------------------------------------------------------------------

class MatchupGraph:
    """
    Bipartite NetworkX graph of NBA offensive–defensive player matchups.

    Node sets:
        bipartite=0  →  offensive players  (node key: ``off_{player_id}``)
        bipartite=1  →  defensive players  (node key: ``def_{player_id}``)

    Edge attributes: weight (PPP), possessions, points, fg_pct
    """

    def __init__(self) -> None:
        self.graph: nx.Graph = nx.Graph()
        self.players: Dict[int, Player] = {}
        self.matchups: Dict[Tuple[int, int], MatchupEdge] = {}
        self.min_possessions: int = 10

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_player(self, player: Player) -> None:
        self.players[player.player_id] = player

    def build_from_dataframe(self, df: pd.DataFrame, min_possessions: int = 10) -> None:
        """Populate the graph from a matchup DataFrame."""
        self.min_possessions = min_possessions
        filtered = df[df["PARTIAL_POSS"] >= min_possessions].copy()

        for _, row in filtered.iterrows():
            off_id = int(row["OFF_PLAYER_ID"])
            def_id = int(row["DEF_PLAYER_ID"])

            for pid, name_col in [(off_id, "OFF_PLAYER_NAME"), (def_id, "DEF_PLAYER_NAME")]:
                if pid not in self.players:
                    self.add_player(Player(pid, row[name_col]))

            def _safe(val, default=0.0):
                try:
                    return float(val) if val is not None and val == val else default
                except (TypeError, ValueError):
                    return default

            edge = MatchupEdge(
                off_player_id=off_id,
                def_player_id=def_id,
                games_played=int(row["GP"]),
                possessions=_safe(row["PARTIAL_POSS"]),
                points=_safe(row["PLAYER_PTS"]),
                fgm=_safe(row["MATCHUP_FGM"]),
                fga=_safe(row["MATCHUP_FGA"]),
                fg_pct=_safe(row.get("MATCHUP_FG_PCT")),
                fg3m=_safe(row.get("MATCHUP_FG3M")),
                fg3a=_safe(row.get("MATCHUP_FG3A")),
                fg3_pct=_safe(row.get("MATCHUP_FG3_PCT")),
                assists=_safe(row.get("MATCHUP_AST")),
                turnovers=_safe(row.get("MATCHUP_TOV")),
                blocks=_safe(row.get("MATCHUP_BLK")),
            )
            self.matchups[(off_id, def_id)] = edge

            off_node, def_node = f"off_{off_id}", f"def_{def_id}"
            self.graph.add_node(off_node, bipartite=0, player_id=off_id,
                                name=row["OFF_PLAYER_NAME"], role="offense")
            self.graph.add_node(def_node, bipartite=1, player_id=def_id,
                                name=row["DEF_PLAYER_NAME"], role="defense")
            self.graph.add_edge(
                off_node, def_node,
                weight=edge.points_per_possession,
                possessions=edge.possessions,
                points=edge.points,
                fg_pct=_safe(row.get("MATCHUP_FG_PCT")),
            )

        self._compute_player_averages()

    def _compute_player_averages(self) -> None:
        for pid, player in self.players.items():
            off_edges = [e for (oi, di), e in self.matchups.items() if oi == pid]
            def_edges = [e for (oi, di), e in self.matchups.items() if di == pid]

            if off_edges:
                tp = sum(e.points for e in off_edges)
                tposs = sum(e.possessions for e in off_edges)
                player.avg_ppp_off = tp / tposs if tposs else None
                player.off_matchup_count = len(off_edges)

            if def_edges:
                tp = sum(e.points for e in def_edges)
                tposs = sum(e.possessions for e in def_edges)
                player.avg_ppp_def = tp / tposs if tposs else None
                player.def_matchup_count = len(def_edges)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def find_player_id(self, name: str) -> Optional[int]:
        """Case-insensitive name lookup; prefers exact match."""
        nl = name.strip().lower()
        exact = [pid for pid, p in self.players.items() if p.name.lower() == nl]
        if exact:
            return exact[0]
        partial = [pid for pid, p in self.players.items() if nl in p.name.lower()]
        return partial[0] if partial else None

    def all_player_names(self, role: Optional[str] = None) -> List[str]:
        names: set = set()
        for (oi, di) in self.matchups:
            if role in (None, "offense") and oi in self.players:
                names.add(self.players[oi].name)
            if role in (None, "defense") and di in self.players:
                names.add(self.players[di].name)
        return sorted(names)

    # ------------------------------------------------------------------
    # Interaction Mode 1 — Matchup Lookup
    # ------------------------------------------------------------------

    def get_matchup(self, off_name: str, def_name: str) -> Optional[MatchupEdge]:
        off_id = self.find_player_id(off_name)
        def_id = self.find_player_id(def_name)
        if off_id is None or def_id is None:
            return None
        return self.matchups.get((off_id, def_id))

    # ------------------------------------------------------------------
    # Interaction Mode 2 — Player Profile / Neighborhood
    # ------------------------------------------------------------------

    def get_offensive_neighborhood(
        self, player_name: str, top_n: int = 999
    ) -> List[Dict]:
        """All matchups as an offensive player, sorted by PPP (descending)."""
        pid = self.find_player_id(player_name)
        if pid is None:
            return []

        rows = []
        for (oi, di), edge in self.matchups.items():
            if oi != pid:
                continue
            defender = self.players.get(di)
            rows.append({
                "defender_id": di,
                "defender": defender.name if defender else str(di),
                "defender_team": defender.team if defender else None,
                "defender_pos": defender.position if defender else None,
                "defender_archetype": (
                    classify_defensive_archetype(defender) if defender else None
                ),
                "ppp": edge.points_per_possession,
                "possessions": edge.possessions,
                "points": edge.points,
                "fg_pct": edge.fg_pct,
                "fg3_pct": edge.fg3_pct,
                "efg_pct": edge.effective_fg_pct,
                "turnovers": edge.turnovers,
                "blocks": edge.blocks,
                "games": edge.games_played,
            })

        rows.sort(key=lambda x: x["ppp"], reverse=True)
        return rows[:top_n]

    def get_defensive_neighborhood(
        self, player_name: str, top_n: int = 999
    ) -> List[Dict]:
        """All matchups as a defensive player, sorted by PPP allowed (ascending)."""
        pid = self.find_player_id(player_name)
        if pid is None:
            return []

        rows = []
        for (oi, di), edge in self.matchups.items():
            if di != pid:
                continue
            scorer = self.players.get(oi)
            rows.append({
                "scorer_id": oi,
                "scorer": scorer.name if scorer else str(oi),
                "scorer_team": scorer.team if scorer else None,
                "scorer_pos": scorer.position if scorer else None,
                "scorer_archetype": (
                    classify_offensive_archetype(scorer) if scorer else None
                ),
                "ppp_allowed": edge.points_per_possession,
                "possessions": edge.possessions,
                "points_allowed": edge.points,
                "fg_pct_allowed": edge.fg_pct,
                "fg3_pct_allowed": edge.fg3_pct,
                "efg_pct_allowed": edge.effective_fg_pct,
                "turnovers_forced": edge.turnovers,
                "blocks": edge.blocks,
                "games": edge.games_played,
            })

        rows.sort(key=lambda x: x["ppp_allowed"])
        return rows[:top_n]

    # ------------------------------------------------------------------
    # Interaction Mode 3 — Similarity (Defenders and Scorers)
    # ------------------------------------------------------------------

    def _build_zscored_stat_vec(
        self,
        pid: int,
        stat_weights: List[Tuple[str, float]],
        pop_stats: Dict[str, Tuple[float, float]],  # stat → (mean, std)
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Build a z-scored stat vector for player pid using precomputed population stats.
        Returns (z_vec, weight_vec) or None if player not found.
        Logs a warning if more than 3 stats are missing.
        """
        p = self.players.get(pid)
        if not p:
            return None
        vec, weights, missing = [], [], 0
        for stat, w in stat_weights:
            mean, std = pop_stats.get(stat, (None, None))
            val = getattr(p, stat, None)
            if val is None or mean is None:
                vec.append(0.0)
                missing += 1
            else:
                vec.append((val - mean) / std)
            weights.append(w)
        if missing > 3:
            logging.warning("Player %d: %d stats missing from tier vector", pid, missing)
        return np.array(vec, dtype=float), np.array(weights, dtype=float)

    @staticmethod
    def _weighted_cosine(va: np.ndarray, wa: np.ndarray,
                         vb: np.ndarray, wb: np.ndarray) -> float:
        """Weighted cosine similarity: scale each element by sqrt(w), then cosine."""
        a = va * np.sqrt(wa)
        b = vb * np.sqrt(wb)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _physical_sim(self, pid_a: int, pid_b: int) -> float:
        """Height/weight/position similarity scaled to [0, 1]."""
        pa, pb = self.players.get(pid_a), self.players.get(pid_b)
        if not pa or not pb:
            return 0.0
        ha = pa.height_inches or 78
        hb = pb.height_inches or 78
        wa = pa.weight or 220
        wb = pb.weight or 220
        h_diff = abs(ha - hb) / 12.0    # normalised over ~1 ft range
        w_diff = abs(wa - wb) / 50.0    # normalised over ~50 lb range
        hw_sim = max(0.0, 1.0 - 0.5 * h_diff - 0.5 * w_diff)
        pos_a = (pa.position or "").upper()
        pos_b = (pb.position or "").upper()
        pos_sim = 1.0 if pos_a == pos_b else (0.5 if pos_a and pos_b and
                                               (pos_a[0] == pos_b[0]) else 0.0)
        return 0.6 * hw_sim + 0.4 * pos_sim

    def find_similar_defenders(
        self, defender_name: str, top_n: int = 10
    ) -> List[Dict]:
        """
        MPS_def = 0.20 × Jaccard(shared opponents)
                + 0.15 × PPP_delta_correlation
                + 0.15 × shot_profile_similarity  (imputed when unavailable)
                + 0.15 × physical_archetype_sim
                + 0.35 × defensive_tier_sim

        defensive_tier_sim = weighted cosine similarity of z-scored stat vectors.
        """
        target_id = self.find_player_id(defender_name)
        if target_id is None:
            return []

        target_opps: Dict[int, float] = {
            oi: e.points_per_possession
            for (oi, di), e in self.matchups.items()
            if di == target_id
        }
        if not target_opps:
            return []

        target_set = set(target_opps)
        all_def_ids = list({di for (_, di) in self.matchups})

        # Stat weights within the 0.35 defensive_tier_sim component
        DEF_TIER = [
            ("epm_def",     0.075),
            ("spg",         0.050),
            ("bpg",         0.045),
            ("avg_ppp_def", 0.035),
            ("p_drb_100",   0.030),
            ("p_stl_100",   0.050),   # remapped SPG proxy — keeps weight total correct
            ("p_blk_100",   0.045),   # remapped BPG proxy
        ]
        # Deduplicate: use per-100 when available, per-game otherwise (stat list is fixed)
        DEF_TIER_FINAL = [
            ("epm_def",     0.075),
            ("p_stl_100",   0.050),
            ("p_blk_100",   0.045),
            ("avg_ppp_def", 0.035),
            ("p_drb_100",   0.030),
        ]
        del DEF_TIER  # not used further

        pop_stats: Dict[str, Tuple[float, float]] = {}
        for stat, _ in DEF_TIER_FINAL:
            vals = [getattr(self.players.get(did), stat, None)
                    for did in all_def_ids if self.players.get(did)]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 3:
                mean, std = float(np.mean(vals)), float(np.std(vals))
                pop_stats[stat] = (mean, max(std, 1e-9))

        target_vec = self._build_zscored_stat_vec(target_id, DEF_TIER_FINAL, pop_stats)

        results = []
        for other_id in all_def_ids:
            if other_id == target_id:
                continue
            other_opps: Dict[int, float] = {
                oi: e.points_per_possession
                for (oi, di), e in self.matchups.items()
                if di == other_id
            }
            shared = target_set & set(other_opps)
            if len(shared) < 3:
                continue

            jaccard = len(shared) / len(target_set | set(other_opps))

            t_vec = np.array([target_opps[o] for o in shared])
            o_vec = np.array([other_opps[o] for o in shared])
            corr = float(np.corrcoef(t_vec, o_vec)[0, 1])
            if np.isnan(corr):
                corr = 0.0
            ppp_delta_corr = (corr + 1) / 2

            phys_sim = self._physical_sim(target_id, other_id)

            other_vec = self._build_zscored_stat_vec(other_id, DEF_TIER_FINAL, pop_stats)
            if target_vec is not None and other_vec is not None:
                tv, tw = target_vec
                ov, ow = other_vec
                raw_cos = self._weighted_cosine(tv, tw, ov, ow)
                tier_sim = (raw_cos + 1) / 2
            else:
                raw_cos = 0.0
                tier_sim = 0.0

            mps_def = (
                0.20 * jaccard
                + 0.15 * ppp_delta_corr
                + 0.15 * 0.0          # shot_profile_sim — imputed until data available
                + 0.15 * phys_sim
                + 0.35 * tier_sim
            )

            op = self.players.get(other_id)
            results.append({
                "defender_id": other_id,
                "defender": op.name if op else str(other_id),
                "team": op.team if op else "—",
                "position": op.position if op else "—",
                "archetype": classify_defensive_archetype(op) if op else None,
                "combined_score": mps_def,
                "mps_def": mps_def,
                "jaccard": jaccard,
                "cosine": raw_cos,           # kept for backward-compat with existing tests
                "correlation": corr,         # kept for backward-compat
                "ppp_delta_corr": ppp_delta_corr,
                "physical_sim": phys_sim,
                "tier_sim": tier_sim,
                "shared_opponents": len(shared),
                "shared_archetype_overlap": 0.0,
                "avg_ppp_def": op.avg_ppp_def if op else None,
            })

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[:top_n]

    def find_similar_scorers(
        self, scorer_name: str, top_n: int = 10
    ) -> List[Dict]:
        """
        MPS_off = 0.20 × Jaccard(shared defenders faced)
                + 0.15 × PPP_delta_correlation
                + 0.15 × shot_zone_similarity  (imputed when unavailable)
                + 0.15 × usage_archetype_sim
                + 0.35 × offensive_tier_sim

        offensive_tier_sim = weighted cosine similarity of z-scored stat vectors.
        """
        target_id = self.find_player_id(scorer_name)
        if target_id is None:
            return []

        target_defs: Dict[int, float] = {
            di: e.points_per_possession
            for (oi, di), e in self.matchups.items()
            if oi == target_id
        }
        if not target_defs:
            return []

        target_set = set(target_defs)
        all_off_ids = list({oi for (oi, _) in self.matchups})

        OFF_TIER = [
            ("ppg",           0.070),
            ("usg_pct",       0.060),
            ("epm_off",       0.055),
            ("ts_pct",        0.035),
            ("ast_pct",       0.030),
            ("p_fga_rim_100", 0.025),
            ("p_fg3a_100",    0.020),
            ("p_fgpct_rim",   0.015),
            ("fg3_pct",       0.015),
            ("apg",           0.010),
            ("ft_pct",        0.005),
            ("p_orb_100",     0.005),
        ]

        pop_stats: Dict[str, Tuple[float, float]] = {}
        for stat, _ in OFF_TIER:
            vals = [getattr(self.players.get(oid), stat, None)
                    for oid in all_off_ids if self.players.get(oid)]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 3:
                mean, std = float(np.mean(vals)), float(np.std(vals))
                pop_stats[stat] = (mean, max(std, 1e-9))

        target_vec = self._build_zscored_stat_vec(target_id, OFF_TIER, pop_stats)
        tp = self.players.get(target_id)

        def _usage_archetype_sim(pid_b: int) -> float:
            pb = self.players.get(pid_b)
            if not tp or not pb:
                return 0.0
            usg_diff = abs((tp.usg_pct or 0.20) - (pb.usg_pct or 0.20)) / 0.30
            ast_diff = abs((tp.ast_pct or 0.10) - (pb.ast_pct or 0.10)) / 0.50
            sim = max(0.0, 1.0 - 0.5 * usg_diff - 0.5 * ast_diff)
            if tp.off_archetype and pb.off_archetype and tp.off_archetype == pb.off_archetype:
                sim = min(1.0, sim + 0.2)
            return sim

        results = []
        for other_id in all_off_ids:
            if other_id == target_id:
                continue
            other_defs: Dict[int, float] = {
                di: e.points_per_possession
                for (oi, di), e in self.matchups.items()
                if oi == other_id
            }
            shared = target_set & set(other_defs)
            if len(shared) < 3:
                continue

            jaccard = len(shared) / len(target_set | set(other_defs))

            t_vec = np.array([target_defs[d] for d in shared])
            o_vec = np.array([other_defs[d] for d in shared])
            corr = float(np.corrcoef(t_vec, o_vec)[0, 1])
            if np.isnan(corr):
                corr = 0.0
            ppp_delta_corr = (corr + 1) / 2

            usage_sim = _usage_archetype_sim(other_id)

            other_vec = self._build_zscored_stat_vec(other_id, OFF_TIER, pop_stats)
            if target_vec is not None and other_vec is not None:
                tv, tw = target_vec
                ov, ow = other_vec
                raw_cos = self._weighted_cosine(tv, tw, ov, ow)
                tier_sim = (raw_cos + 1) / 2
            else:
                raw_cos = 0.0
                tier_sim = 0.0

            mps_off = (
                0.20 * jaccard
                + 0.15 * ppp_delta_corr
                + 0.15 * 0.0       # shot_zone_sim — imputed until zone data stored on Player
                + 0.15 * usage_sim
                + 0.35 * tier_sim
            )

            op = self.players.get(other_id)
            results.append({
                "scorer_id": other_id,
                "scorer": op.name if op else str(other_id),
                "team": op.team if op else "—",
                "position": op.position if op else "—",
                "archetype": classify_offensive_archetype(op) if op else None,
                "combined_score": mps_off,
                "mps_off": mps_off,
                "jaccard": jaccard,
                "cosine": raw_cos,
                "ppp_delta_corr": ppp_delta_corr,
                "usage_sim": usage_sim,
                "tier_sim": tier_sim,
                "shared_opponents": len(shared),
                "avg_ppp_off": op.avg_ppp_off if op else None,
            })

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[:top_n]

    # ------------------------------------------------------------------
    # Graph Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict:
        off_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("role") == "offense"]
        def_nodes = [n for n, d in self.graph.nodes(data=True) if d.get("role") == "defense"]
        degrees = [d for _, d in self.graph.degree()]
        ppps = [e.points_per_possession for e in self.matchups.values()]
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "offensive_players": len(off_nodes),
            "defensive_players": len(def_nodes),
            "total_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "avg_degree": float(np.mean(degrees)) if degrees else 0,
            "avg_ppp": float(np.mean(ppps)) if ppps else 0,
            "min_possessions": self.min_possessions,
        }

    def degree_sequences(self) -> Tuple[List[int], List[int]]:
        off_degs = [d for n, d in self.graph.degree()
                    if self.graph.nodes[n].get("role") == "offense"]
        def_degs = [d for n, d in self.graph.degree()
                    if self.graph.nodes[n].get("role") == "defense"]
        return off_degs, def_degs

    def top_connected(self, role: str, top_n: int = 10) -> List[Dict]:
        """Most connected players by degree."""
        nodes = [(n, d) for n, d in self.graph.degree()
                 if self.graph.nodes[n].get("role") == role]
        nodes.sort(key=lambda x: x[1], reverse=True)
        results = []
        for node, deg in nodes[:top_n]:
            pid = self.graph.nodes[node]["player_id"]
            player = self.players.get(pid)
            results.append({
                "name": self.graph.nodes[node]["name"],
                "team": player.team if player else "—",
                "position": player.position if player else "—",
                "connections": deg,
            })
        return results
