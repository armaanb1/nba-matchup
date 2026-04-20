"""
Professional Plotly visualizations for the NBA Matchup Network.

Color palette (aligned with app.py CSS):
  Blue   #3b82f6   (primary / Team 1)
  Amber  #f59e0b   (secondary / CounterPoint accent)
  Green  #10b981   (positive / better-than-reputation)
  Red    #ef4444   (negative / worse-than-reputation)
  Gray   #475569   (neutral)
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
NAVY = "#3b82f6"       # primary blue (replaces #1D428A)
RED = "#ef4444"        # negative red
GOLD = "#f59e0b"       # amber accent
GREEN = "#10b981"      # positive green
LIGHT_GRAY = "#94a3b8"
DARK_BG = "#0a0e17"
CARD_BG = "#131a2b"
FONT_COLOR = "#f1f5f9"
BORDER_COLOR = "#1e293b"
MUTED = "#475569"

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=DARK_BG,
    font=dict(family="'DM Sans', 'Inter', Arial, sans-serif", size=12, color=FONT_COLOR),
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
) -> "pd.DataFrame":
    """
    Returns a color-coded DataFrame comparing target defender to top-k similar defenders.
    Green = better than target, red = worse. lower_is_better stats are inverted.
    """
    import pandas as pd

    # (label, getter, lower_is_better)
    stats_config = [
        ("PPP Allowed", lambda p: p.avg_ppp_def or 0,                          True),
        ("Matchups",    lambda p: p.def_matchup_count,                          False),
        ("DEPM",        lambda p: getattr(p, "epm_def", None) or 0,            False),
        ("BPG",         lambda p: p.bpg or 0,                                  False),
        ("SPG",         lambda p: p.spg or 0,                                  False),
        ("BLK/100",     lambda p: getattr(p, "p_blk_100", None) or 0,         False),
        ("STL/100",     lambda p: getattr(p, "p_stl_100", None) or 0,         False),
    ]

    players = [target]
    labels = [target.name]
    for sim in similar_list[:top_k]:
        other = graph_obj.players.get(sim["defender_id"])
        if other:
            players.append(other)
            labels.append(f"{other.name} ({sim['combined_score']:.2f})")

    stat_names = [s[0] for s in stats_config]
    data = {}
    for p, label in zip(players, labels):
        data[label] = [s[1](p) for s in stats_config]

    df = pd.DataFrame(data, index=stat_names)

    def _style(row):
        stat = row.name
        config = next((s for s in stats_config if s[0] == stat), None)
        lower_is_better = config[2] if config else False
        target_val = row.iloc[0]
        styles = [""]  # target column unstyled
        for val in row.iloc[1:]:
            if val == 0 and target_val == 0:
                styles.append("")
            elif (lower_is_better and val < target_val) or (not lower_is_better and val > target_val):
                styles.append("background-color: rgba(34,197,94,0.3)")
            elif (lower_is_better and val > target_val) or (not lower_is_better and val < target_val):
                styles.append("background-color: rgba(239,68,68,0.3)")
            else:
                styles.append("")
        return styles

    # Format each row's decimal places individually
    fmt_map = {
        "PPP Allowed": "{:.3f}",
        "Matchups":    "{:.0f}",
    }
    default_fmt = "{:.1f}"

    format_dict = {}
    for col in df.columns:
        format_dict[col] = {stat: fmt_map.get(stat, default_fmt) for stat in df.index}

    # Build a per-cell format dict keyed by (col, stat)
    cell_fmt = {}
    for col in df.columns:
        for stat in df.index:
            cell_fmt[(stat, col)] = fmt_map.get(stat, default_fmt)

    # Apply format per stat (row) across all columns
    row_formats = {stat: fmt_map.get(stat, default_fmt) for stat in df.index}

    styled = df.style.apply(_style, axis=1)
    for stat, fmt in row_formats.items():
        styled = styled.format(fmt, subset=pd.IndexSlice[stat, :])
    return styled


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
    Center-aligned diverging horizontal bar chart comparing two teams.
    Team 1 bars extend LEFT from center (blue), Team 2 bars extend RIGHT (amber).
    The team with the statistical advantage on each row is highlighted; the other
    is rendered in a muted slate so the winner stands out instantly.
    Lower-is-better stats are flipped so longer bar always = better.
    """
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

    rows: list = []
    for label, key, invert, as_pct in stat_defs:
        v1 = t1.get(key)
        v2 = t2.get(key)
        if v1 is None or v2 is None:
            continue
        v1f, v2f = float(v1), float(v2)
        t1_txt = f"{v1f:.1%}" if as_pct else f"{v1f:.1f}"
        t2_txt = f"{v2f:.1%}" if as_pct else f"{v2f:.1f}"
        if as_pct:
            v1f, v2f = v1f * 100, v2f * 100
        if invert:
            v1f, v2f = -v1f, -v2f
        rows.append((label, v1f, v2f, t1_txt, t2_txt))

    if not rows:
        return go.Figure()

    labels    = [r[0] for r in rows]
    # For the diverging layout: t1 extends left (negative), t2 extends right (positive)
    # Normalise each row so the better score = ±1.0 and the worse = a fraction of that
    left_vals, right_vals = [], []
    left_cols, right_cols = [], []
    left_texts, right_texts = [], []

    for label, v1, v2, t1_txt, t2_txt in rows:
        mx = max(abs(v1), abs(v2), 1e-9)
        # Negative = left (Team 1), Positive = right (Team 2)
        lv = -(abs(v1) / mx)
        rv =  (abs(v2) / mx)
        left_vals.append(lv)
        right_vals.append(rv)
        left_texts.append(t1_txt)
        right_texts.append(t2_txt)
        # Highlight the winner; mute the loser
        t1_wins = v1 >= v2
        left_cols.append(NAVY if t1_wins else MUTED)
        right_cols.append(GOLD if not t1_wins else MUTED)

    fig = go.Figure()

    # Team 1 — left bars
    fig.add_trace(go.Bar(
        name=team1,
        y=labels, x=left_vals,
        orientation="h",
        marker_color=left_cols,
        text=left_texts,
        textposition="outside",
        textfont=dict(size=11, color=FONT_COLOR),
        hovertemplate="%{text}<extra>" + team1 + "</extra>",
    ))

    # Team 2 — right bars
    fig.add_trace(go.Bar(
        name=team2,
        y=labels, x=right_vals,
        orientation="h",
        marker_color=right_cols,
        text=right_texts,
        textposition="outside",
        textfont=dict(size=11, color=FONT_COLOR),
        hovertemplate="%{text}<extra>" + team2 + "</extra>",
    ))

    fig.update_layout(
        barmode="overlay",
        height=520,
        xaxis=dict(
            showgrid=False, showticklabels=False, zeroline=True,
            zerolinecolor=BORDER_COLOR, zerolinewidth=2,
            range=[-1.55, 1.55],
        ),
        yaxis=dict(
            gridcolor=BORDER_COLOR,
            tickfont=dict(size=12, color=FONT_COLOR, family="'DM Sans', Arial"),
        ),
        legend=dict(
            bgcolor=CARD_BG, bordercolor=BORDER_COLOR, borderwidth=1,
            font=dict(color=FONT_COLOR, size=13),
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        ),
        annotations=[
            dict(
                x=-0.78, y=1.04, xref="paper", yref="paper",
                text=f"◀  {team1}", showarrow=False,
                font=dict(size=12, color=NAVY),
            ),
            dict(
                x=0.78, y=1.04, xref="paper", yref="paper",
                text=f"{team2}  ▶", showarrow=False,
                font=dict(size=12, color=GOLD),
            ),
        ],
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


# ---------------------------------------------------------------------------
# CounterPoint — sparkline trend chart
# ---------------------------------------------------------------------------

def plot_sparkline(
    seasons: List[str],
    values: List[float],
    stat_label: str,
    flag: str,
    height: int = 90,
) -> go.Figure:
    """
    Small sparkline showing a single stat's trajectory across seasons.
    Color-coded by CounterPoint flag type:
      better_than_reputation → green
      worse_than_reputation  → red
      role_shift             → gold
    """
    color_map = {
        "better_than_reputation": "#00875A",
        "worse_than_reputation":  "#C8102E",
        "role_shift":             "#F0A500",
    }
    color = color_map.get(flag, "#9CA3AF")

    # Shorten season labels for readability (e.g. "2023-24" → "'23")
    short = [s.split("-")[0][-2:] + "/" + s.split("-")[1][-2:] if "-" in s else s
             for s in seasons]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=short,
        y=values,
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=5, color=color),
        showlegend=False,
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ))

    # Subtle fill under the line — convert #RRGGBB to rgba(r,g,b,0.12)
    def _hex_to_rgba(h: str, a: float = 0.12) -> str:
        h = h.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{a})"

    fig.add_trace(go.Scatter(
        x=short,
        y=values,
        fill="tozeroy",
        mode="none",
        fillcolor=_hex_to_rgba(color) if color.startswith("#") else color,
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=4, r=4, t=18, b=4),
        paper_bgcolor="#1A2035",
        plot_bgcolor="#1A2035",
        xaxis=dict(
            showticklabels=True,
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=8, color="#9CA3AF"),
        ),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        title=dict(
            text=stat_label,
            font=dict(size=9, color="#9CA3AF"),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Shot Chart
# ---------------------------------------------------------------------------

def _draw_court_shapes() -> List[Dict]:
    """Return a list of Plotly shape dicts for a half-court outline."""
    import math

    shapes = []

    def arc_points(cx, cy, r, angle_start_deg, angle_end_deg, n=60):
        angles = np.linspace(math.radians(angle_start_deg), math.radians(angle_end_deg), n)
        xs = [cx + r * math.cos(a) for a in angles]
        ys = [cy + r * math.sin(a) for a in angles]
        return xs, ys

    # Court boundary (half court)
    shapes.append(dict(
        type="rect", x0=-250, y0=-47.5, x1=250, y1=422.5,
        line=dict(color=BORDER_COLOR, width=1.5), fillcolor="rgba(0,0,0,0)",
    ))

    # Paint (key) — outer box
    shapes.append(dict(
        type="rect", x0=-80, y0=-47.5, x1=80, y1=142.5,
        line=dict(color=BORDER_COLOR, width=1.5), fillcolor="rgba(19,26,43,0.6)",
    ))

    # Free throw lane (inner)
    shapes.append(dict(
        type="rect", x0=-60, y0=-47.5, x1=60, y1=142.5,
        line=dict(color=BORDER_COLOR, width=1), fillcolor="rgba(0,0,0,0)",
    ))

    # Free throw circle (top half)
    ft_xs, ft_ys = arc_points(0, 142.5, 60, 0, 180)
    shapes.append(dict(
        type="path",
        path="M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in zip(ft_xs, ft_ys)),
        line=dict(color=BORDER_COLOR, width=1.5), fillcolor="rgba(0,0,0,0)",
    ))

    # Restricted area arc (~40 inches = ~40 units radius from basket)
    ra_xs, ra_ys = arc_points(0, 0, 40, 0, 180)
    shapes.append(dict(
        type="path",
        path="M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in zip(ra_xs, ra_ys)),
        line=dict(color=BORDER_COLOR, width=1.5), fillcolor="rgba(0,0,0,0)",
    ))

    # 3-point arc — straight side lines + arc
    # Side 3-pt lines
    shapes.append(dict(type="line", x0=-220, y0=-47.5, x1=-220, y1=92.5,
                       line=dict(color=BORDER_COLOR, width=1.5)))
    shapes.append(dict(type="line", x0=220, y0=-47.5, x1=220, y1=92.5,
                       line=dict(color=BORDER_COLOR, width=1.5)))

    # 3-point arc (radius ~237.5 from basket at 0,0)
    tp_xs, tp_ys = arc_points(0, 0, 237.5, 22.0, 158.0)
    shapes.append(dict(
        type="path",
        path="M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in zip(tp_xs, tp_ys)),
        line=dict(color=BORDER_COLOR, width=1.5), fillcolor="rgba(0,0,0,0)",
    ))

    # Basket backboard
    shapes.append(dict(type="line", x0=-30, y0=-7.5, x1=30, y1=-7.5,
                       line=dict(color=FONT_COLOR, width=2)))

    return shapes


_ZONE_COLOR_MAP = {
    "Restricted Area":         "#10b981",   # green — high efficiency
    "In The Paint (Non-RA)":   "#f59e0b",   # amber
    "Mid-Range":               "#ef4444",   # red — low efficiency
    "Left Corner 3":           "#3b82f6",   # blue
    "Right Corner 3":          "#3b82f6",   # blue
    "Above the Break 3":       "#a855f7",   # purple
    "Backcourt":               "#475569",   # muted
}


def plot_shot_chart(shot_df, player_name: str) -> go.Figure:
    """
    Scatter shot chart on a half-court diagram.

    shot_df must have columns: LOC_X, LOC_Y, SHOT_MADE_FLAG, SHOT_ZONE_BASIC
    Made shots = filled circle; missed shots = open X marker.
    Each zone gets a distinct color.
    """
    if shot_df.empty or "LOC_X" not in shot_df.columns:
        fig = go.Figure()
        fig.update_layout(
            title="No shot chart data available",
            **LAYOUT_DEFAULTS,
        )
        return fig

    made = shot_df[shot_df["SHOT_MADE_FLAG"] == 1]
    missed = shot_df[shot_df["SHOT_MADE_FLAG"] == 0]

    traces = []

    # Made shots — filled circles, colored by zone
    for zone, grp in made.groupby("SHOT_ZONE_BASIC"):
        color = _ZONE_COLOR_MAP.get(zone, "#94a3b8")
        traces.append(go.Scatter(
            x=grp["LOC_X"], y=grp["LOC_Y"],
            mode="markers",
            name=f"{zone} (made)",
            legendgroup=zone,
            marker=dict(symbol="circle", size=5, color=color, opacity=0.75,
                        line=dict(width=0)),
            hovertemplate=(
                f"<b>{zone}</b><br>Made<br>(%{{x}}, %{{y}})<extra></extra>"
            ),
        ))

    # Missed shots — x markers, same zone color but more transparent
    for zone, grp in missed.groupby("SHOT_ZONE_BASIC"):
        color = _ZONE_COLOR_MAP.get(zone, "#94a3b8")
        traces.append(go.Scatter(
            x=grp["LOC_X"], y=grp["LOC_Y"],
            mode="markers",
            name=f"{zone} (missed)",
            legendgroup=zone,
            showlegend=False,
            marker=dict(symbol="x", size=5, color=color, opacity=0.35,
                        line=dict(width=1)),
            hovertemplate=(
                f"<b>{zone}</b><br>Missed<br>(%{{x}}, %{{y}})<extra></extra>"
            ),
        ))

    # Basket marker
    traces.append(go.Scatter(
        x=[0], y=[0], mode="markers",
        marker=dict(symbol="circle", size=10, color=FONT_COLOR,
                    line=dict(color=DARK_BG, width=2)),
        name="Basket", showlegend=False,
        hoverinfo="skip",
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        shapes=_draw_court_shapes(),
        xaxis=dict(range=[-260, 260], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-50, 430], showgrid=False, zeroline=False,
                   showticklabels=False),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
            font=dict(size=10, color=FONT_COLOR),
            bgcolor="rgba(0,0,0,0)",
            itemsizing="constant",
        ),
        height=520,
        **LAYOUT_DEFAULTS,
    )
    _clean_layout(fig, title=f"{player_name} — Shot Chart")
    return fig
