"""
Core data models for the NBA Matchup Network.

Classes:
    Player         — graph node representing an NBA player
    MatchupEdge    — weighted edge between offensive and defensive player
    MatchupGraph   — bipartite NetworkX graph with all four interaction modes
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd


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

    # Matchup-derived (computed from graph)
    avg_ppp_off: Optional[float] = None   # avg PPP scored on offense
    avg_ppp_def: Optional[float] = None   # avg PPP allowed on defense
    off_matchup_count: int = 0
    def_matchup_count: int = 0

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
    # Interaction Mode 3 — Defensive Similarity
    # ------------------------------------------------------------------

    def find_similar_defenders(
        self, defender_name: str, top_n: int = 10
    ) -> List[Dict]:
        """
        Graph-based defensive similarity:
          - Jaccard similarity (shared offensive opponents)
          - Cosine similarity of PPP vectors over shared opponents
          - Pearson correlation over shared opponents
        Combined = 0.4 × Jaccard + 0.3 × Cosine + 0.3 × (Corr + 1) / 2
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
        all_defenders = {di for (_, di) in self.matchups}

        results = []
        for other_id in all_defenders:
            if other_id == target_id:
                continue
            other_opps: Dict[int, float] = {
                oi: e.points_per_possession
                for (oi, di), e in self.matchups.items()
                if di == other_id
            }
            other_set = set(other_opps)
            shared = target_set & other_set
            if len(shared) < 3:
                continue

            jaccard = len(shared) / len(target_set | other_set)
            t_vec = np.array([target_opps[o] for o in shared])
            o_vec = np.array([other_opps[o] for o in shared])

            nt, no = np.linalg.norm(t_vec), np.linalg.norm(o_vec)
            cosine = float(np.dot(t_vec, o_vec) / (nt * no)) if nt and no else 0.0

            corr = float(np.corrcoef(t_vec, o_vec)[0, 1])
            if np.isnan(corr):
                corr = 0.0

            combined = 0.4 * jaccard + 0.3 * cosine + 0.3 * (corr + 1) / 2

            other_player = self.players.get(other_id)
            results.append({
                "defender_id": other_id,
                "defender": other_player.name if other_player else str(other_id),
                "team": other_player.team if other_player else "—",
                "position": other_player.position if other_player else "—",
                "combined_score": combined,
                "jaccard": jaccard,
                "cosine": cosine,
                "correlation": corr,
                "shared_opponents": len(shared),
                "avg_ppp_def": other_player.avg_ppp_def if other_player else None,
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
