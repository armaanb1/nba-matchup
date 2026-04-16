"""
Professional Plotly visualizations for the NBA Matchup Network.

Color palette:
  Navy   #1D428A   (primary nodes / bars)
  Red    #C8102E   (opponent nodes / warning)
  Gold   #F0A500   (center player / highlight)
  Green  #00875A   (good defense)
  Gray   #6B7280   (neutral)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models import MatchupGraph, Player

# --------------------------------------------------
# Design constants
# --------------------------------------------------
NAVY = "#1D428A"
RED = "#C8102E"
GOLD = "#F0A500"
GREEN = "#00875A"
LIGHT_GRAY = "#F8F9FA"
DARK_BG = "#0E1117"
CARD_BG = "#1A2035"
FONT_COLOR = "#FAFAFA"

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_BG,
    font=dict(family="Inter, Helvetica, Arial", size=12, color=FONT_COLOR),
    margin=dict(l=40, r=40, t=60, b=40),
)


def _clean_layout(fig: go.Figure, title: str = "", **kwargs) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=16, color=FONT_COLOR)),
                      **LAYOUT_DEFAULTS, **kwargs)
    return fig


# ---------------------------------------------------------------------------
# 1. Matchup Comparison Bar Chart (Mode 1)
# ---------------------------------------------------------------------------

def plot_matchup_comparison(
    matchup_dict: Dict,
    off_player: Player,
    def_player: Player,
) -> go.Figure:
    """
    Side-by-side bar chart comparing the matchup stats to each player's averages.
    matchup_dict should contain a 'PPP' key (from MatchupEdge.to_dict()).
    """
    # to_dict() uses capitalized keys; fall back to lowercase for flexibility
    ppp_raw = matchup_dict.get("PPP") or matchup_dict.get("ppp", "0")
    ppp = float(ppp_raw) if isinstance(ppp_raw, str) else (ppp_raw or 0)
    off_avg = off_player.avg_ppp_off or 0
    def_avg = def_player.avg_ppp_def or 0

    categories = ["PPP in\nthis matchup", f"{off_player.name.split()[-1]}\navg PPP (off)",
                  f"{def_player.name.split()[-1]}\navg PPP allowed (def)"]
    values = [ppp, off_avg, def_avg]
    colors = [GOLD, NAVY, RED]

    fig = go.Figure(go.Bar(
        x=categories, y=values, marker_color=colors,
        text=[f"{v:.3f}" for v in values], textposition="outside",
        textfont=dict(size=13, color=FONT_COLOR),
    ))

    fig.add_hline(y=1.0, line_dash="dot", line_color="#6B7280",
                  annotation_text="League avg ~1.0 PPP",
                  annotation_position="top right",
                  annotation_font=dict(color="#6B7280"))

    fig.update_yaxes(range=[0, max(values) * 1.35],
                     gridcolor="#2A3550", zerolinecolor="#2A3550")
    fig.update_xaxes(showgrid=False)
    return _clean_layout(fig,
                         title=f"Points Per Possession — {off_player.name} vs {def_player.name}",
                         showlegend=False)


# ---------------------------------------------------------------------------
# 2. Player Neighborhood — Bar Charts (Mode 2)
# ---------------------------------------------------------------------------

def plot_neighborhood_bars(
    rows: List[Dict],
    player_name: str,
    role: str,
    top_n: int = 12,
) -> go.Figure:
    """
    Horizontal bar chart: best (top) and worst (bottom) matchups for a player.
    """
    if not rows:
        return go.Figure()

    ppp_key = "ppp" if role == "offense" else "ppp_allowed"
    opp_key = "defender" if role == "offense" else "scorer"
    label = "PPP scored" if role == "offense" else "PPP allowed"

    rows_sorted = sorted(rows, key=lambda x: x[ppp_key], reverse=(role == "offense"))
    best = rows_sorted[:top_n]
    worst = rows_sorted[-top_n:][::-1]

    def _bars(data, bar_color, title_txt):
        names = [r[opp_key].split(" ", 1)[-1] for r in data]   # last name
        ppps = [r[ppp_key] for r in data]
        poss = [r["possessions"] for r in data]

        return go.Bar(
            y=names, x=ppps,
            orientation="h",
            marker=dict(
                color=ppps,
                colorscale=[[0, GREEN], [0.5, GOLD], [1, RED]] if role == "offense"
                            else [[0, GREEN], [0.5, GOLD], [1, RED]],
                showscale=False,
            ),
            text=[f"{p:.3f} | {int(po)}p" for p, po in zip(ppps, poss)],
            textposition="outside",
            textfont=dict(size=11, color=FONT_COLOR),
            name=title_txt,
        )

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[
                            f"Best Matchups (Highest {label})" if role == "offense"
                            else "Easiest Matchups (Lowest PPP allowed)",
                            f"Toughest Matchups (Lowest {label})" if role == "offense"
                            else "Hardest Matchups (Highest PPP allowed)",
                        ])

    fig.add_trace(_bars(best, GREEN if role == "defense" else RED, "Best"), row=1, col=1)
    fig.add_trace(_bars(worst, RED if role == "defense" else GREEN, "Worst"), row=1, col=2)

    fig.update_xaxes(gridcolor="#2A3550", zerolinecolor="#2A3550")
    fig.update_yaxes(showgrid=False)
    fig.update_layout(height=420, showlegend=False, **LAYOUT_DEFAULTS,
                      title=dict(text=f"{player_name} — Matchup Breakdown ({role.title()})",
                                 font=dict(size=15, color=FONT_COLOR)))
    return fig


# ---------------------------------------------------------------------------
# 3. Interactive Network Graph (Mode 2 / overview)
# ---------------------------------------------------------------------------

def plot_network_neighborhood(
    graph_obj: MatchupGraph,
    player_name: str,
    role: str = "offense",
    top_n: int = 18,
) -> go.Figure:
    """
    Interactive Plotly network graph of a player's matchup neighborhood.
    Nodes are sized by possessions; edges colored by PPP.
    """
    pid = graph_obj.find_player_id(player_name)
    if pid is None:
        return go.Figure()

    center_node = f"{role[:3]}_{pid}"
    if center_node not in graph_obj.graph:
        return go.Figure()

    # Select top-N neighbors by possessions
    neighbor_data = []
    for nb in graph_obj.graph.neighbors(center_node):
        ed = graph_obj.graph.edges[center_node, nb]
        neighbor_data.append((nb, ed["possessions"], ed["weight"]))
    neighbor_data.sort(key=lambda x: x[1], reverse=True)
    top_neighbors = [n for n, _, _ in neighbor_data[:top_n]]

    sub_nodes = [center_node] + top_neighbors
    subg = graph_obj.graph.subgraph(sub_nodes)

    # Layout
    pos = nx.kamada_kawai_layout(subg)

    # Build edge traces (colored by PPP)
    edge_x, edge_y, edge_hover = [], [], []
    edge_ppps = []
    for u, v in subg.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        ppp = subg.edges[u, v]["weight"]
        edge_ppps.append(ppp)

    # Color edges by quartile
    if edge_ppps:
        q1, q3 = np.percentile(edge_ppps, 25), np.percentile(edge_ppps, 75)

    def _ppp_color(ppp):
        if not edge_ppps:
            return "#888888"
        norm = (ppp - min(edge_ppps)) / (max(edge_ppps) - min(edge_ppps) + 1e-9)
        r = int(norm * 200 + 55)
        g = int((1 - norm) * 180 + 30)
        return f"rgb({r},{g},80)"

    # Build node traces
    node_x, node_y, node_text, node_color, node_size, node_hover = [], [], [], [], [], []
    for node in subg.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        nd = subg.nodes[node]
        name = nd.get("name", node)
        is_center = (node == center_node)

        # Size by possessions-degree
        if is_center:
            size = 34
            color = GOLD
        elif nd.get("role") == "offense":
            size = 20
            color = NAVY
        else:
            size = 20
            color = RED

        node_size.append(size)
        node_color.append(color)

        last = name.split()[-1] if name else node
        node_text.append(name if is_center else last)

        # Hover text
        team = nd.get("team", "")
        pos_txt = nd.get("position", "")
        node_hover.append(
            f"<b>{name}</b><br>Role: {nd.get('role','')}<br>"
            f"Team: {team}<br>Pos: {pos_txt}"
        )

    # One edge trace per edge (for individual coloring)
    edge_traces = []
    for idx, (u, v) in enumerate(subg.edges()):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ppp = subg.edges[u, v]["weight"]
        poss = subg.edges[u, v]["possessions"]
        color = _ppp_color(ppp)
        width = max(1.5, min(6, poss / 25))

        off_name = subg.nodes[u]["name"] if subg.nodes[u].get("role") == "offense" else subg.nodes[v]["name"]
        def_name = subg.nodes[v]["name"] if subg.nodes[v].get("role") == "defense" else subg.nodes[u]["name"]

        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="text",
            hovertext=f"{off_name} vs {def_name}<br>PPP: {ppp:.3f} | Poss: {int(poss)}",
            showlegend=False,
        ))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(color=node_color, size=node_size, line=dict(color="#FFFFFF", width=2)),
        text=node_text,
        textposition="top center",
        textfont=dict(size=12, color=FONT_COLOR, family="Inter, Arial, sans-serif"),
        hovertext=node_hover,
        hoverinfo="text",
        showlegend=False,
    )

    # Legend annotation
    annotations = [
        dict(x=0.01, y=0.99, xref="paper", yref="paper", showarrow=False,
             text=(
                 "<b style='color:#F0A500'>●</b> Gold = selected player&nbsp;&nbsp;"
                 "<b style='color:#4A90D9'>■</b> Blue = offensive node&nbsp;&nbsp;"
                 "<b style='color:#E05252'>■</b> Red = defensive node<br>"
                 "Edge width = possessions&nbsp;&nbsp;|&nbsp;&nbsp;"
                 "Edge color: <span style='color:#37c83c'><b>green</b> = low PPP</span>"
                 " → <span style='color:#e07030'><b>orange/red</b> = high PPP</span>"
             ),
             font=dict(size=12, color=FONT_COLOR), align="left",
             bgcolor=CARD_BG, bordercolor="#6B7280", borderwidth=1),
    ]

    fig = go.Figure(data=[*edge_traces, node_trace])
    fig.update_layout(
        title=dict(text=f"{player_name} — {role.title()} Matchup Network (top {top_n} by possessions)",
                   font=dict(size=15, color=FONT_COLOR)),
        annotations=annotations,
        height=550,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        **LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Defensive Similarity Radar (Mode 3)
# ---------------------------------------------------------------------------

def plot_similarity_comparison(
    target: Player,
    similar_list: List[Dict],
    graph_obj: MatchupGraph,
    top_k: int = 3,
) -> tuple:
    """
    Returns (table_df, bar_fig) comparing target defender to top-k similar defenders.
    Table shows raw stats side by side; bar chart groups by stat for easy comparison.
    """
    stats_config = [
        ("PPP Allowed", lambda p: round((p.avg_ppp_def or 0), 3)),
        ("Matchups",    lambda p: p.def_matchup_count),
        ("DEPM",        lambda p: round(getattr(p, "epm_def", None) or 0, 2)),
        ("BPG",         lambda p: round(p.bpg or 0, 1)),
        ("SPG",         lambda p: round(p.spg or 0, 1)),
        ("Def Rating",  lambda p: round(p.def_rating or 0, 1)),
    ]

    players = [target]
    labels = [target.name]
    for sim in similar_list[:top_k]:
        other = graph_obj.players.get(sim["defender_id"])
        if other:
            players.append(other)
            labels.append(f"{other.name} ({sim['combined_score']:.2f})")

    # Build comparison table
    rows = {"Stat": [s[0] for s in stats_config]}
    for p, label in zip(players, labels):
        rows[label] = [s[1](p) for s in stats_config]
    import pandas as pd
    table_df = pd.DataFrame(rows)

    # Build grouped bar chart
    colors = [GOLD, RED, NAVY, GREEN, "#A855F7"]
    fig = go.Figure()
    stat_names = [s[0] for s in stats_config]

    for idx, (p, label) in enumerate(zip(players, labels)):
        vals = [s[1](p) for s in stats_config]
        fig.add_trace(go.Bar(
            name=label,
            x=stat_names,
            y=vals,
            marker_color=colors[idx % len(colors)],
            text=[str(v) for v in vals],
            textposition="outside",
            textfont=dict(size=10, color=FONT_COLOR),
        ))

    fig.update_layout(
        barmode="group",
        bargap=0.2,
        bargroupgap=0.05,
        xaxis=dict(tickfont=dict(color=FONT_COLOR), gridcolor="#2A3550"),
        yaxis=dict(tickfont=dict(color=FONT_COLOR), gridcolor="#2A3550", title="Value"),
        legend=dict(bgcolor=CARD_BG, bordercolor="#374151", borderwidth=1,
                    font=dict(color=FONT_COLOR)),
        height=380,
        title=dict(text=f"Defensive Profile — {target.name} vs Similar Defenders",
                   font=dict(size=14, color=FONT_COLOR)),
        **LAYOUT_DEFAULTS,
    )
    return table_df, fig


def _hex_to_rgb(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"{r},{g},{b}"


# ---------------------------------------------------------------------------
# 5. Similarity Bar Chart (Mode 3)
# ---------------------------------------------------------------------------

def plot_similarity_scores(similar_list: List[Dict], target_name: str) -> go.Figure:
    if not similar_list:
        return go.Figure()

    names = [s["defender"] for s in similar_list]
    combined = [s["combined_score"] for s in similar_list]
    jaccard = [s["jaccard"] for s in similar_list]
    cosine = [s["cosine"] for s in similar_list]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Combined Score", x=names, y=combined, marker_color=GOLD, offsetgroup=0,
        text=[f"{v:.2f}" for v in combined], textposition="outside",
        textfont=dict(size=12, color=FONT_COLOR),
    ))
    fig.add_trace(go.Bar(
        name="Jaccard (shared opponents)", x=names, y=jaccard, marker_color=NAVY, offsetgroup=1,
        text=[f"{v:.2f}" for v in jaccard], textposition="outside",
        textfont=dict(size=12, color=FONT_COLOR),
    ))
    fig.add_trace(go.Bar(
        name="Cosine (PPP pattern)", x=names, y=cosine, marker_color=RED, offsetgroup=2,
        text=[f"{v:.2f}" for v in cosine], textposition="outside",
        textfont=dict(size=12, color=FONT_COLOR),
    ))

    fig.update_layout(
        barmode="group",
        xaxis=dict(
            tickangle=-35, gridcolor="#2A3550",
            tickfont=dict(size=13, color=FONT_COLOR),
        ),
        yaxis=dict(
            title=dict(text="Similarity Score", font=dict(size=13, color=FONT_COLOR)),
            range=[0, 1.15], gridcolor="#2A3550",
            tickfont=dict(size=12, color=FONT_COLOR),
        ),
        legend=dict(
            bgcolor=CARD_BG, bordercolor="#6B7280", borderwidth=1,
            font=dict(color=FONT_COLOR, size=14),
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
        height=440,
        title=dict(text=f"Defenders Most Similar to {target_name}",
                   font=dict(size=15, color=FONT_COLOR)),
        **LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Degree Distribution (Graph Overview)
# ---------------------------------------------------------------------------

def plot_degree_distribution(off_degs: List[int], def_degs: List[int]) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Offensive Players — Defenders Faced",
                                        "Defensive Players — Scorers Guarded"])

    fig.add_trace(go.Histogram(x=off_degs, nbinsx=30,
                                marker_color=NAVY, name="Offense",
                                marker_line=dict(color="#1a2744", width=0.5)), row=1, col=1)
    fig.add_trace(go.Histogram(x=def_degs, nbinsx=30,
                                marker_color=RED, name="Defense",
                                marker_line=dict(color="#1a2744", width=0.5)), row=1, col=2)

    fig.update_xaxes(gridcolor="#2A3550", zerolinecolor="#2A3550")
    fig.update_yaxes(gridcolor="#2A3550", zerolinecolor="#2A3550")
    fig.update_layout(height=360, showlegend=False,
                      title=dict(text="Degree Distribution — Matchup Graph",
                                 font=dict(size=14, color=FONT_COLOR)),
                      **LAYOUT_DEFAULTS)
    return fig


# ---------------------------------------------------------------------------
# 7. PPP Heatmap for top players (Graph Overview)
# ---------------------------------------------------------------------------

def plot_ppp_heatmap(graph_obj: MatchupGraph, top_n: int = 15) -> go.Figure:
    """
    Heatmap of PPP for top offensive players (rows) × top defensive players (cols).
    """
    # Pick top offensive players by total possessions
    off_poss: Dict[int, float] = {}
    def_poss: Dict[int, float] = {}
    for (oi, di), e in graph_obj.matchups.items():
        off_poss[oi] = off_poss.get(oi, 0) + e.possessions
        def_poss[di] = def_poss.get(di, 0) + e.possessions

    top_off = sorted(off_poss, key=off_poss.get, reverse=True)[:top_n]
    top_def = sorted(def_poss, key=def_poss.get, reverse=True)[:top_n]

    off_names = [graph_obj.players[i].name.split()[-1] for i in top_off if i in graph_obj.players]
    def_names = [graph_obj.players[i].name.split()[-1] for i in top_def if i in graph_obj.players]

    # Build matrix
    matrix = np.full((len(top_off), len(top_def)), np.nan)
    for ri, oi in enumerate(top_off):
        for ci, di in enumerate(top_def):
            edge = graph_obj.matchups.get((oi, di))
            if edge:
                matrix[ri, ci] = edge.points_per_possession

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=def_names, y=off_names,
        colorscale=[[0, GREEN], [0.5, GOLD], [1, RED]],
        zmid=1.0,
        colorbar=dict(title="PPP", tickfont=dict(color=FONT_COLOR)),
        hovertemplate="Off: %{y}<br>Def: %{x}<br>PPP: %{z:.3f}<extra></extra>",
        text=[[f"{v:.2f}" if not np.isnan(v) else "" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont=dict(size=8),
    ))

    fig.update_layout(
        title=dict(text=f"PPP Heatmap — Top {top_n} Offensive vs Defensive Players",
                   font=dict(size=14, color=FONT_COLOR)),
        xaxis=dict(title="Defensive Player", tickangle=-40, showgrid=False),
        yaxis=dict(title="Offensive Player", showgrid=False),
        height=500,
        **LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# 8. Player stat bar chart (Mode 2 — bio card supplement)
# ---------------------------------------------------------------------------

def plot_team_comparison_bars(t1: dict, t2: dict, team1: str, team2: str) -> go.Figure:
    """
    Grouped horizontal bar chart comparing two teams across key stats.
    Lower-is-better stats (Def Rtg, TOV%) are inverted so longer bar = better.
    Team1 bars in NAVY, Team2 bars in RED.
    """
    # (label, t1_key, t2_key, invert, format_pct)
    stat_defs = [
        ("Off Rtg",  "OFF_RATING",  False, False),
        ("Def Rtg",  "DEF_RATING",  True,  False),
        ("Net Rtg",  "NET_RATING",  False, False),
        ("Pace",     "PACE",        False, False),
        ("eFG%",     "EFG_PCT",     False, True),
        ("TOV%",     "TM_TOV_PCT",  True,  True),
        ("OReb%",    "OREB_PCT",    False, True),
        ("TS%",      "TS_PCT",      False, True),
    ]

    labels, vals1, vals2, texts1, texts2 = [], [], [], [], []

    for label, key, invert, as_pct in stat_defs:
        v1 = t1.get(key)
        v2 = t2.get(key)
        if v1 is None or v2 is None:
            continue

        v1f, v2f = float(v1), float(v2)

        if as_pct:
            t1_txt = f"{v1f:.1%}"
            t2_txt = f"{v2f:.1%}"
            # Scale to 0-100 for consistent bar sizing with non-pct stats
            v1f_display = v1f * 100
            v2f_display = v2f * 100
        else:
            t1_txt = f"{v1f:.1f}"
            t2_txt = f"{v2f:.1f}"
            v1f_display = v1f
            v2f_display = v2f

        if invert:
            # Flip so higher bar = better (lower raw value is better)
            max_val = max(v1f_display, v2f_display) + 1e-9
            v1f_display = max_val - v1f_display + max_val * 0.1
            v2f_display = max_val - v2f_display + max_val * 0.1

        labels.append(label)
        vals1.append(v1f_display)
        vals2.append(v2f_display)
        texts1.append(t1_txt)
        texts2.append(t2_txt)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=team1,
        y=labels, x=vals1,
        orientation="h",
        marker_color=NAVY,
        text=texts1,
        textposition="outside",
        textfont=dict(size=11, color=FONT_COLOR),
        offsetgroup=0,
    ))
    fig.add_trace(go.Bar(
        name=team2,
        y=labels, x=vals2,
        orientation="h",
        marker_color=RED,
        text=texts2,
        textposition="outside",
        textfont=dict(size=11, color=FONT_COLOR),
        offsetgroup=1,
    ))

    fig.update_layout(
        barmode="group",
        height=500,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(gridcolor="#2A3550", tickfont=dict(size=12, color=FONT_COLOR)),
        legend=dict(
            bgcolor=CARD_BG, bordercolor="#374151", borderwidth=1,
            font=dict(color=FONT_COLOR, size=13),
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
        title=dict(text=f"Team Comparison — {team1} vs {team2}",
                   font=dict(size=15, color=FONT_COLOR)),
        **LAYOUT_DEFAULTS,
    )
    return fig


def plot_team_radar(t1: dict, t2: dict, team1: str, team2: str) -> go.Figure:
    """
    Radar/spider chart comparing two teams across normalized stat dimensions.
    Team1 in GOLD, Team2 in RED. Each stat normalized 0-1 across the two teams.
    Lower-is-better stats (Def Rtg, TOV%) are inverted before normalization.
    """
    # (label, key, invert)
    stat_defs = [
        ("Off Rtg",  "OFF_RATING", False),
        ("Def Rtg",  "DEF_RATING", True),
        ("Net Rtg",  "NET_RATING", False),
        ("eFG%",     "EFG_PCT",    False),
        ("TOV%",     "TM_TOV_PCT", True),
        ("OReb%",    "OREB_PCT",   False),
    ]

    categories = []
    norm1, norm2 = [], []

    for label, key, invert in stat_defs:
        v1 = t1.get(key)
        v2 = t2.get(key)
        if v1 is None or v2 is None:
            continue

        v1f, v2f = float(v1), float(v2)
        if invert:
            v1f, v2f = -v1f, -v2f

        lo = min(v1f, v2f)
        hi = max(v1f, v2f)
        rng = hi - lo if hi != lo else 1e-9
        categories.append(label)
        norm1.append((v1f - lo) / rng)
        norm2.append((v2f - lo) / rng)

    if not categories:
        return go.Figure()

    theta = categories + [categories[0]]
    r1 = norm1 + [norm1[0]]
    r2 = norm2 + [norm2[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r1, theta=theta,
        fill="toself",
        fillcolor=f"rgba({_hex_to_rgb(GOLD)},0.18)",
        line=dict(color=GOLD, width=2.5),
        name=team1,
    ))
    fig.add_trace(go.Scatterpolar(
        r=r2, theta=theta,
        fill="toself",
        fillcolor=f"rgba({_hex_to_rgb(RED)},0.15)",
        line=dict(color=RED, width=2.5),
        name=team2,
    ))

    fig.update_layout(
        polar=dict(
            bgcolor=CARD_BG,
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor="#2A3550", color="#6B7280",
                tickfont=dict(size=8), showticklabels=False,
            ),
            angularaxis=dict(gridcolor="#2A3550", color=FONT_COLOR,
                             tickfont=dict(size=11)),
        ),
        legend=dict(
            bgcolor=CARD_BG, bordercolor="#374151", borderwidth=1,
            font=dict(color=FONT_COLOR, size=13),
        ),
        height=420,
        title=dict(text=f"Team Radar — {team1} vs {team2}",
                   font=dict(size=14, color=FONT_COLOR)),
        **LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# 8. Player stat bar chart (Mode 2 — bio card supplement)
# ---------------------------------------------------------------------------

def plot_player_stats_bar(player: Player) -> go.Figure:
    """Quick glance bar chart of per-game stats."""
    stats = {
        "PPG": player.ppg,
        "RPG": player.rpg,
        "APG": player.apg,
        "SPG": player.spg,
        "BPG": player.bpg,
        "TOV": player.tov,
    }
    labels = [k for k, v in stats.items() if v is not None]
    values = [v for v in stats.values() if v is not None]

    if not labels:
        return go.Figure()

    colors = [GOLD if l == "PPG" else NAVY if l in ("RPG", "APG") else RED for l in labels]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
        textfont=dict(size=13, color=FONT_COLOR),
    ))
    fig.update_yaxes(gridcolor="#2A3550", range=[0, max(values) * 1.35])
    fig.update_xaxes(showgrid=False)
    layout = {**LAYOUT_DEFAULTS, "margin": dict(l=30, r=30, t=50, b=20)}
    fig.update_layout(showlegend=False, height=260,
                      title=dict(text="Per-Game Stats", font=dict(size=13, color=FONT_COLOR)),
                      **layout)
    return fig
