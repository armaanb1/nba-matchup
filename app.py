"""
NBA Offensive–Defensive Player Matchup Network
SI 507 Final Project — Streamlit Application

Four interaction modes:
  1. Matchup Lookup        — head-to-head stats for any offensive vs. defensive player pair
  2. Player Profile        — full matchup neighborhood + bio + Basketball Reference stats
  3. Defensive Similarity  — find defenders with the most similar matchup profiles
  4. LLM Scouting Report  — Anthropic-powered narrative scouting reports
"""
import numpy as np
import pandas as pd
import streamlit as st

from data_loader import (
    enrich_graph,
    get_playoff_series,
    get_team_roster,
    get_team_standings_live,
    get_team_stats_live,
    load_matchup_data,
)
from llm_reports import (
    generate_matchup_report,
    generate_player_profile_report,
    generate_playoff_matchup_keys,
    generate_similarity_report,
    generate_team_matchup_report,
)
from models import MatchupGraph
from visualizations import (
    plot_degree_distribution,
    plot_matchup_comparison,
    plot_neighborhood_bars,
    plot_network_neighborhood,
    plot_player_stats_bar,
    plot_ppp_heatmap,
    plot_similarity_comparison,
    plot_similarity_scores,
    plot_sparkline,
    plot_team_comparison_bars,
    plot_team_radar,
)
from counterpoint import (
    FLAG_COLOR,
    FLAG_LABEL,
    STAT_LABELS as CP_STAT_LABELS,
    call_cp_briefing,
    call_cp_qa,
    compute_drift,
    generate_example_questions,
    get_cross_team_matchups,
)
from data_loader import get_player_career_splits

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NBA Matchup Network",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Base ---- */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background-color: #1A2035;
        border-right: 1px solid #2A3550;
    }
    [data-testid="stSidebar"] * { color: #FAFAFA !important; }

    /* ---- Sidebar selectbox values: always black text ---- */
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: #111111 !important; }
    /* ---- Metric cards ---- */
    [data-testid="stMetric"] {
        background: #1A2035;
        border: 1px solid #2A3550;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; color: #F0A500 !important; }
    [data-testid="stMetricLabel"] { color: #9CA3AF !important; font-size: 0.78rem !important; }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #1A2035;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #9CA3AF;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: #1D428A !important;
        color: #FFFFFF !important;
    }

    /* ---- Data tables ---- */
    .dataframe-container { border-radius: 8px; overflow: hidden; }
    [data-testid="stDataFrame"] { border: 1px solid #2A3550; border-radius: 8px; }

    /* ---- Stat card ---- */
    .stat-card {
        background: #1A2035;
        border: 1px solid #2A3550;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }
    .stat-card h4 { color: #F0A500; margin: 0 0 8px 0; font-size: 0.85rem; letter-spacing: 0.08em; text-transform: uppercase; }
    .stat-card p { margin: 2px 0; color: #D1D5DB; font-size: 0.92rem; }
    .stat-card span.value { color: #FAFAFA; font-weight: 600; }

    /* ---- Report output ---- */
    .report-box {
        background: #1A2035;
        border: 1px solid #2A3550;
        border-left: 4px solid #F0A500;
        border-radius: 10px;
        padding: 20px 24px;
        line-height: 1.7;
        font-size: 0.95rem;
        color: #E5E7EB;
        white-space: pre-wrap;
    }

    /* ---- Section header ---- */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #F0A500;
        border-bottom: 1px solid #2A3550;
        padding-bottom: 6px;
        margin: 18px 0 12px 0;
    }

    /* ---- Player name badge ---- */
    .player-badge {
        display: inline-block;
        background: #1D428A;
        color: #FAFAFA;
        border-radius: 6px;
        padding: 4px 12px;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 8px;
    }

    /* ---- Info box ---- */
    .info-box {
        background: #162032;
        border: 1px solid #2A3550;
        border-radius: 8px;
        padding: 12px 16px;
        color: #9CA3AF;
        font-size: 0.88rem;
    }

    /* ---- Divider ---- */
    hr { border-color: #2A3550 !important; }

    /* Remove Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* ---- CounterPoint ---- */
    .cp-entry {
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 10px;
    }
    .cp-entry .cp-player-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #FAFAFA;
        margin-bottom: 6px;
    }
    .cp-entry .cp-narrative  { color: #9CA3AF; font-size: 0.88rem; margin: 3px 0; }
    .cp-entry .cp-numbers    { font-size: 0.88rem; margin: 3px 0; font-weight: 600; }
    .cp-entry .cp-coaching   { color: #D1D5DB; font-size: 0.88rem; margin: 3px 0; font-style: italic; }
    .cp-entry .cp-link       { color: #F0A500; font-size: 0.82rem; margin-top: 6px; }

    .cp-flag-callout {
        background: #162032;
        border: 1px solid #2A3550;
        border-left: 4px solid #F0A500;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 10px 0;
        font-size: 0.88rem;
    }
    .cp-badge {
        display: inline-block;
        background: #F0A500;
        color: #0E1117;
        font-weight: 800;
        font-size: 0.72rem;
        border-radius: 4px;
        padding: 2px 6px;
        margin-right: 6px;
        letter-spacing: 0.04em;
        vertical-align: middle;
    }
    .cp-briefing {
        background: #162032;
        border: 1px solid #2A3550;
        border-left: 4px solid #1D428A;
        border-radius: 10px;
        padding: 20px 24px;
        line-height: 1.75;
        font-size: 0.95rem;
        color: #E5E7EB;
        white-space: pre-wrap;
    }
    .cp-response-card {
        background: #1A2035;
        border: 1px solid #2A3550;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 8px;
    }
    .cp-response-header {
        font-size: 0.75rem;
        font-weight: 700;
        color: #F0A500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .cp-leaderboard-row {
        background: #1A2035;
        border: 1px solid #2A3550;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "graph": None,
        "data_loaded": False,
        "enriched": False,
        "season": "2025-26",
        "season_type": "Regular Season",
        "min_poss": 20,
        "api_key": "",
        "team_stats_df": None,
        "standings_df": None,
        "playoff_series_df": None,
        "team_data_loaded": False,
        "team_data_updated_at": None,
        "roster_cache": {},          # {team_id: DataFrame}
        "roster_team_ids": {},       # {team_name: team_id}
        # CounterPoint state
        "cp_player_drift": {},       # {player_id: drift_dict | None}
        "cp_matchup_drift": {},      # {player_id: drift_dict | None} for selected CP matchup
        "cp_leaderboard_drift": {},  # {player_id: drift_dict | None} for leaderboard
        "cp_chat_history": [],       # [{role, content}, ...] up to 6 messages
        "cp_briefing": "",           # current generated briefing text
        "cp_team1": "",
        "cp_team2": "",
        "cp_nav_player": None,       # player name pre-loaded from Scouting Report link
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

graph: MatchupGraph | None = st.session_state.graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stat_card(title: str, items: dict) -> str:
    rows = "".join(
        f'<p>{k}: <span class="value">{v}</span></p>'
        for k, v in items.items()
        if v not in (None, "—", "")
    )
    return f'<div class="stat-card"><h4>{title}</h4>{rows}</div>'


def _ppp_color(ppp: float, avg: float = 1.0) -> str:
    if ppp > avg + 0.15:
        return "#C8102E"
    if ppp < avg - 0.15:
        return "#00875A"
    return "#F0A500"


def _get_or_compute_drift(player_id: int, player_name: str = "") -> dict | None:
    """
    Return cached drift result for player_id, computing it if not yet cached.
    Stores result in st.session_state.cp_player_drift[player_id].
    Returns None if insufficient career history or player below threshold.
    """
    cache = st.session_state.cp_player_drift
    if player_id in cache:
        return cache[player_id]
    try:
        splits_df = get_player_career_splits(player_id)
        result = compute_drift(player_id, splits_df, st.session_state.season, player_name=player_name)
    except Exception:
        result = None
    cache[player_id] = result
    return result


def _render_cp_flag(player_id: int, player_name: str) -> None:
    """
    Show a compact CounterPoint flag callout for a player if a drift flag
    exists above threshold.  Does nothing when no flag is present.
    """
    drift = _get_or_compute_drift(player_id, player_name)
    if drift is None or not drift.get("flagged"):
        return

    flag  = drift["flag"]
    stat  = drift["max_drift_stat"]
    color = FLAG_COLOR.get(flag, "#F0A500")
    label = FLAG_LABEL.get(flag, "Flag")
    slbl  = CP_STAT_LABELS.get(stat, stat)

    career_v = drift["career_avgs"].get(stat)
    curr_v   = drift["current_vals"].get(stat)
    fmt = ".1%" if "pct" in stat or stat == "ft_rate" else ".1f"
    career_str = f"{career_v:{fmt}}" if career_v is not None else "—"
    curr_str   = f"{curr_v:{fmt}}"   if curr_v  is not None else "—"
    arrow      = "↑" if (curr_v or 0) > (career_v or 0) else "↓"

    # Store the player name so CounterPoint tab can pre-load it
    if st.session_state.cp_nav_player is None:
        st.session_state.cp_nav_player = player_name

    st.markdown(
        f'<div class="cp-flag-callout">'
        f'<span class="cp-badge">CP</span>'
        f'<span style="color:{color}; font-weight:700;">{label}</span>'
        f'&nbsp;&nbsp;'
        f'<span style="color:#D1D5DB;">{slbl}: {career_str} {arrow} {curr_str}</span>'
        f'<br>'
        f'<span style="color:#9CA3AF;">{drift["coaching_impl"]}</span>'
        f'<br>'
        f'<span style="color:#F0A500; font-size:0.82rem;">Full analysis in CounterPoint →</span>'
        f'</div>',
        unsafe_allow_html=True,
    )



# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🏀 NBA Matchup Network")
    st.markdown("---")

    st.markdown("### Data Settings")
    season = st.selectbox("Season", ["2025-26", "2024-25", "2023-24"], index=0)
    season_type = st.selectbox("Season Type", ["Regular Season", "Playoffs"], index=0)
    min_poss = st.slider("Min Possessions per Matchup", 5, 50, 20, step=5)

    st.markdown("---")
    st.markdown("### Anthropic API Key")
    api_key_input = st.text_input(
        "API Key (for Scouting Reports)",
        value=st.session_state.api_key,
        type="password",
        placeholder="sk-ant-...",
    )
    if api_key_input:
        st.session_state.api_key = api_key_input

    st.markdown("---")

    load_btn = st.button("⬇ Load Data", use_container_width=True, type="primary")
    enrich_btn = st.button("🔄 Refresh Data", use_container_width=True,
                           type="primary",
                           help="Pull latest bio + advanced stats for all players (includes playoff data)",
                           disabled=not st.session_state.data_loaded)
    st.caption("Baseline data: April 16, 2026")

    # Load matchup data
    if load_btn:
        with st.spinner(f"Loading {season} matchup data…"):
            try:
                df = load_matchup_data(season, season_type, min_possessions=min_poss)
                g = MatchupGraph()
                g.build_from_dataframe(df, min_possessions=min_poss)
                st.session_state.graph = g
                st.session_state.data_loaded = True
                st.session_state.enriched = False
                st.session_state.season = season
                st.session_state.season_type = season_type
                st.session_state.min_poss = min_poss
                graph = g
                st.success(f"Graph built — {g.graph.number_of_nodes()} nodes, "
                           f"{g.graph.number_of_edges()} edges")
                # Auto-enrich from cache (fast since cache files are pre-committed)
                try:
                    enrich_graph(g, season=season)
                    st.session_state.enriched = True
                    graph = g
                except Exception:
                    pass  # silently skip — user can manually refresh
            except Exception as e:
                st.error(f"Load failed: {e}")

    # Refresh player data
    if enrich_btn and st.session_state.graph:
        prog_bar = st.progress(0, text="Refreshing players…")
        total = len(st.session_state.graph.players)

        def _prog(i, tot, name):
            prog_bar.progress(i / tot, text=f"Refreshing {name}… ({i}/{tot})")

        with st.spinner("Fetching latest bio + stats for all players…"):
            try:
                enrich_graph(st.session_state.graph,
                             season=st.session_state.season,
                             progress_callback=_prog,
                             force_refresh_epm=True)
                st.session_state.enriched = True
                graph = st.session_state.graph
                prog_bar.empty()
                st.success("Player data refreshed!")
            except Exception as e:
                prog_bar.empty()
                st.error(f"Refresh error: {e}")

    # Graph summary in sidebar
    if st.session_state.data_loaded and st.session_state.graph:
        g = st.session_state.graph
        summ = g.get_summary()
        st.markdown("---")
        st.markdown("### Graph Summary")
        st.metric("Players (Offense)", summ["offensive_players"])
        st.metric("Players (Defense)", summ["defensive_players"])
        st.metric("Matchup Edges", summ["total_edges"])
        st.metric("Avg Connections", f"{summ['avg_degree']:.1f}")
        st.metric("Avg PPP", f"{summ['avg_ppp']:.3f}")
        enriched_status = "✅ Enriched" if st.session_state.enriched else "⚠ Basic only"
        st.caption(enriched_status)

    graph = st.session_state.graph


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

# Hero header
st.markdown(
    """
    <div style="text-align:center; padding: 24px 0 8px 0;">
        <h1 style="font-size:2.4rem; font-weight:800; color:#FAFAFA; margin:0;">
            🏀 NBA Matchup Network
        </h1>
        <p style="color:#9CA3AF; font-size:1.05rem; margin:6px 0 0 0;">
            Bipartite graph analysis of NBA offensive–defensive player matchups
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

if not st.session_state.data_loaded or graph is None:
    st.markdown(
        """
        <div class="info-box" style="text-align:center; padding:40px; font-size:1.05rem;">
            <b style="color:#F0A500; font-size:1.2rem;">Get Started</b><br><br>
            Use the sidebar to load NBA matchup data for a season.<br>
            The first load pulls from the NBA Stats API and caches results locally.<br><br>
            <span style="color:#6B7280;">Typical first-load time: 30–60 seconds</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔍  Matchup Lookup",
    "👤  Player Profile",
    "🛡  Defensive Similarity",
    "🤖  Scouting Report",
    "📊  Graph Overview",
    "🏆  Team Matchup",
    "🎯  CounterPoint",
])


# ===========================================================================
# TAB 1 — Matchup Lookup
# ===========================================================================
with tab1:
    st.markdown('<div class="section-header">Matchup Lookup</div>', unsafe_allow_html=True)
    st.markdown("Search any offensive–defensive player pair for their head-to-head stats.")

    all_names = sorted(set(graph.all_player_names("offense")) | set(graph.all_player_names("defense")))
    off_names = graph.all_player_names("offense")
    def_names = graph.all_player_names("defense")

    col1, col2 = st.columns(2)
    with col1:
        off_sel = st.selectbox("Offensive Player", off_names,
                               index=off_names.index("LeBron James") if "LeBron James" in off_names else 0,
                               key="ml_off")
    with col2:
        def_sel = st.selectbox("Defensive Player", def_names,
                               index=0, key="ml_def")

    lookup_btn = st.button("Look Up Matchup", type="primary")

    if lookup_btn or (off_sel and def_sel):
        edge = graph.get_matchup(off_sel, def_sel)
        off_pid = graph.find_player_id(off_sel)
        def_pid = graph.find_player_id(def_sel)
        off_player = graph.players.get(off_pid) if off_pid else None
        def_player = graph.players.get(def_pid) if def_pid else None

        if edge and off_player and def_player:
            st.markdown("---")

            # Header
            st.markdown(
                f'<div style="text-align:center; font-size:1.4rem; font-weight:700; color:#FAFAFA; margin-bottom:12px;">'
                f'<span class="player-badge">{off_player.name}</span>'
                f'<span style="color:#9CA3AF; margin: 0 12px;">vs</span>'
                f'<span class="player-badge">{def_player.name}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Key metrics
            avg_ppp = graph.get_summary()["avg_ppp"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Points Per Possession", f"{edge.points_per_possession:.3f}",
                      delta=f"{edge.points_per_possession - avg_ppp:+.3f} vs league avg")
            c2.metric("Possessions", f"{edge.possessions:.0f}")
            c3.metric("Points Scored", f"{edge.points:.0f}")
            c4.metric("FG%", f"{edge.fg_pct:.1%}")
            c5.metric("eFG%", f"{edge.effective_fg_pct:.1%}")

            c6, c7, c8, c9 = st.columns(4)
            c6.metric("3PT Made / Att", f"{edge.fg3m:.0f}/{edge.fg3a:.0f}")
            c7.metric("Assists", f"{edge.assists:.1f}")
            c8.metric("Turnovers", f"{edge.turnovers:.1f}")
            c9.metric("Blocks", f"{edge.blocks:.1f}")

            st.markdown("---")

            # Comparison chart
            col_chart, col_context = st.columns([2, 1])
            with col_chart:
                st.plotly_chart(
                    plot_matchup_comparison(edge.to_dict(), off_player, def_player),
                    use_container_width=True,
                )

            with col_context:
                st.markdown("**Context**")
                if off_player.avg_ppp_off:
                    diff_off = edge.points_per_possession - off_player.avg_ppp_off
                    arrow = "▲" if diff_off > 0 else "▼"
                    color = "#C8102E" if diff_off > 0 else "#00875A"
                    st.markdown(
                        f"<p style='color:{color};'>{arrow} {abs(diff_off):.3f} PPP vs "
                        f"{off_player.name.split()[-1]}'s average</p>",
                        unsafe_allow_html=True,
                    )
                if def_player.avg_ppp_def:
                    diff_def = edge.points_per_possession - def_player.avg_ppp_def
                    arrow = "▲" if diff_def > 0 else "▼"
                    color = "#C8102E" if diff_def > 0 else "#00875A"
                    st.markdown(
                        f"<p style='color:{color};'>{arrow} {abs(diff_def):.3f} PPP vs "
                        f"{def_player.name.split()[-1]}'s average allowed</p>",
                        unsafe_allow_html=True,
                    )

                if off_player.ppg:
                    st.markdown(
                        f"<p style='color:#9CA3AF; font-size:0.85rem;'>"
                        f"{off_player.name.split()[-1]} scores {off_player.ppg:.1f} PPG this season.</p>",
                        unsafe_allow_html=True,
                    )
                if def_player.bpg or def_player.spg:
                    blocks_txt = f"{def_player.bpg:.1f} BPG" if def_player.bpg else ""
                    steals_txt = f"{def_player.spg:.1f} SPG" if def_player.spg else ""
                    sep = ", " if blocks_txt and steals_txt else ""
                    st.markdown(
                        f"<p style='color:#9CA3AF; font-size:0.85rem;'>"
                        f"{def_player.name.split()[-1]} averages {blocks_txt}{sep}{steals_txt}.</p>",
                        unsafe_allow_html=True,
                    )

            # Full stats table
            with st.expander("Full Matchup Stats Table"):
                st.dataframe(
                    pd.DataFrame([edge.to_dict()]).T.rename(columns={0: "Value"}),
                    use_container_width=True,
                )
        else:
            if not off_player:
                st.warning(f"'{off_sel}' not found in offensive player set.")
            elif not def_player:
                st.warning(f"'{def_sel}' not found in defensive player set.")
            else:
                st.info(
                    f"No direct matchup found between **{off_sel}** and **{def_sel}** "
                    f"with ≥{graph.min_possessions} possessions this season."
                )


# ===========================================================================
# TAB 2 — Player Profile
# ===========================================================================

# Stat field mapping: display name -> (Player attribute, lower_is_better)
_PG_STAT_FIELDS = {
    "PPG": ("ppg", False), "RPG": ("rpg", False), "APG": ("apg", False),
    "SPG": ("spg", False), "BPG": ("bpg", False), "TOV": ("tov", True),
    "MPG": ("mpg", False), "FG%": ("fg_pct", False), "3P%": ("fg3_pct", False),
    "FT%": ("ft_pct", False), "TS%": ("ts_pct", False),
}
_ADV_STAT_FIELDS = {
    "Off Rating": ("off_rating", False), "Def Rating": ("def_rating", True),
    "Net Rating": ("net_rating", False), "USG%": ("usg_pct", False),
    "PIE": ("pie", False), "AST%": ("ast_pct", False),
    "EPM": ("epm_tot", False), "OEPM": ("epm_off", False), "DEPM": ("epm_def", False),
    "PTS/100": ("p_pts_100", False), "AST/100": ("p_ast_100", False),
    "BLK/100": ("p_blk_100", False), "STL/100": ("p_stl_100", False),
    "DRB/100": ("p_drb_100", False), "ORB/100": ("p_orb_100", False),
    "TOV/100": ("p_tov_100", True),
    "Rim FGA/100": ("p_fga_rim_100", False), "Mid FGA/100": ("p_fga_mid_100", False),
    "3PA/100": ("p_fg3a_100", False),
    "Rim FG%": ("p_fgpct_rim", False), "Mid FG%": ("p_fgpct_mid", False),
}

def _pct_label(value, all_values, lower_is_better=False):
    """Return 'XXth' percentile string for value among all_values."""
    vals = [v for v in all_values if v is not None]
    if not vals or value is None:
        return "—"
    pct = float(np.mean(np.array(vals) <= value)) * 100
    if lower_is_better:
        pct = 100 - pct
    n = round(pct)
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

def _build_stat_df(player, stat_fields, all_players):
    """Build a stat dataframe with Value and Pct columns."""
    rows = []
    for label, (field, lower) in stat_fields.items():
        val = getattr(player, field, None)
        if val is None:
            continue
        all_vals = [getattr(p, field, None) for p in all_players]
        fmt = f"{val:.1%}" if "pct" in field or field == "usg_pct" or field == "ast_pct" else f"{val:.1f}"
        rows.append({"Stat": label, "Value": fmt, "Pct": _pct_label(val, all_vals, lower)})
    return pd.DataFrame(rows)
with tab2:
    st.markdown('<div class="section-header">Player Profile</div>', unsafe_allow_html=True)
    st.markdown("Explore a player's full matchup neighborhood, stats, and graph centrality.")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        player_sel = st.selectbox("Player", graph.all_player_names(),
                                  index=0, key="pp_player")
    with col_b:
        role_sel = st.radio("Role", ["offense", "defense"], horizontal=True, key="pp_role")

    if player_sel:
        pid = graph.find_player_id(player_sel)
        player = graph.players.get(pid) if pid else None

        if not player:
            st.warning("Player not found in graph.")
            st.stop()

        st.markdown("---")

        # ---- Bio + stats row ----
        bio_col, stats_col, adv_col = st.columns([1.2, 1.4, 1.2])

        with bio_col:
            st.markdown(
                f'<div class="player-badge">{player.name}</div>',
                unsafe_allow_html=True,
            )
            bio = player.bio_dict()
            st.markdown(_stat_card("Player Bio", bio), unsafe_allow_html=True)

            if player.avg_ppp_off is not None:
                st.markdown(
                    _stat_card("Matchup Profile", {
                        "Avg PPP (offense)": f"{player.avg_ppp_off:.3f}",
                        "Avg PPP allowed (defense)": f"{player.avg_ppp_def:.3f}" if player.avg_ppp_def else "—",
                        "Offensive matchups": player.off_matchup_count,
                        "Defensive matchups": player.def_matchup_count,
                    }),
                    unsafe_allow_html=True,
                )

        with stats_col:
            st.markdown('<h4 style="color:#F0A500; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em;">Per-Game Stats</h4>', unsafe_allow_html=True)
            all_players = list(graph.players.values())
            pg_df = _build_stat_df(player, _PG_STAT_FIELDS, all_players)
            if not pg_df.empty:
                st.dataframe(pg_df, hide_index=True, use_container_width=True, height=310)

            if any(v is not None for v in [player.ppg, player.rpg, player.apg, player.spg, player.bpg]):
                st.plotly_chart(plot_player_stats_bar(player), use_container_width=True)

        with adv_col:
            st.markdown('<h4 style="color:#F0A500; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em;">Advanced Stats</h4>', unsafe_allow_html=True)
            if st.session_state.enriched:
                adv_df = _build_stat_df(player, _ADV_STAT_FIELDS, all_players)
                if adv_df.empty:
                    st.markdown(
                        '<div class="info-box">No advanced stats available for this player.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.dataframe(adv_df, hide_index=True, use_container_width=True, height=200)
            else:
                st.markdown(
                    '<div class="info-box">Advanced stats not yet loaded.<br>'
                    'Click <b>Refresh Data</b> in the sidebar.</div>',
                    unsafe_allow_html=True,
                )

            # Matchup-derived averages as metrics — always show when data is available
            if role_sel == "offense" and player.avg_ppp_off is not None:
                league_avg = graph.get_summary()["avg_ppp"]
                st.metric("Avg PPP scored", f"{player.avg_ppp_off:.3f}",
                          delta=f"{player.avg_ppp_off - league_avg:+.3f} vs lg avg")
            elif role_sel == "defense" and player.avg_ppp_def is not None:
                league_avg = graph.get_summary()["avg_ppp"]
                st.metric("Avg PPP allowed", f"{player.avg_ppp_def:.3f}",
                          delta=f"{player.avg_ppp_def - league_avg:+.3f} vs lg avg",
                          delta_color="inverse")

        st.markdown("---")

        # ---- Neighborhood ----
        st.markdown(
            f'<div class="section-header">{player.name} — {role_sel.title()} Neighborhood</div>',
            unsafe_allow_html=True,
        )

        if role_sel == "offense":
            neighborhood = graph.get_offensive_neighborhood(player.name)
        else:
            neighborhood = graph.get_defensive_neighborhood(player.name)

        if not neighborhood:
            st.info(f"No matchup data found for {player.name} as a {role_sel}.")
        else:
            n1, n2 = st.columns([1.2, 1])

            with n1:
                st.plotly_chart(
                    plot_neighborhood_bars(neighborhood, player.name, role_sel),
                    use_container_width=True,
                )

            with n2:
                st.plotly_chart(
                    plot_network_neighborhood(graph, player.name, role_sel, top_n=16),
                    use_container_width=True,
                )

            # Full table
            with st.expander(f"All {len(neighborhood)} matchups — full table"):
                if role_sel == "offense":
                    disp = [
                        {
                            "Defender": r["defender"],
                            "Team": r.get("defender_team") or "—",
                            "PPP": f"{r['ppp']:.3f}",
                            "Poss": f"{r['possessions']:.0f}",
                            "Pts": f"{r['points']:.0f}",
                            "FG%": f"{r['fg_pct']:.1%}",
                            "eFG%": f"{r['efg_pct']:.1%}",
                            "TOV": f"{r['turnovers']:.1f}",
                            "BLK": f"{r['blocks']:.1f}",
                        }
                        for r in neighborhood
                    ]
                else:
                    disp = [
                        {
                            "Scorer": r["scorer"],
                            "Team": r.get("scorer_team") or "—",
                            "PPP Allowed": f"{r['ppp_allowed']:.3f}",
                            "Poss": f"{r['possessions']:.0f}",
                            "Pts Allowed": f"{r['points_allowed']:.0f}",
                            "FG% Allowed": f"{r['fg_pct_allowed']:.1%}",
                            "TOV Forced": f"{r['turnovers_forced']:.1f}",
                            "BLK": f"{r['blocks']:.1f}",
                        }
                        for r in neighborhood
                    ]
                st.dataframe(pd.DataFrame(disp), hide_index=True, use_container_width=True)


# ===========================================================================
# TAB 3 — Defensive Similarity
# ===========================================================================
with tab3:
    st.markdown('<div class="section-header">Defensive Similarity</div>', unsafe_allow_html=True)
    st.markdown(
        "Find defenders with the most similar matchup profiles — they lock down and struggle "
        "against the same sets of offensive players. Useful for trade evaluation, defensive "
        "assignments, and scouting replacements."
    )

    def_names_all = graph.all_player_names("defense")
    def_sel3 = st.selectbox("Select a Defender", def_names_all, key="ds_def")
    top_n3 = st.slider("Show top N similar defenders", 3, 15, 8)

    sim_btn = st.button("Find Similar Defenders", type="primary")

    if sim_btn and def_sel3:
        with st.spinner("Computing graph-based similarity…"):
            similar = graph.find_similar_defenders(def_sel3, top_n=top_n3)

        def_pid = graph.find_player_id(def_sel3)
        def_player = graph.players.get(def_pid) if def_pid else None

        if not similar:
            st.warning("Not enough shared opponents to compute similarity. "
                       "Try a player with more matchup data or lower the min possessions.")
        else:
            st.markdown("---")

            # Target player header
            if def_player:
                meta_parts = filter(None, [
                    def_player.position, def_player.team,
                    f"{def_player.height}" if def_player.height else None,
                    f"{def_player.weight} lbs" if def_player.weight else None,
                ])
                st.markdown(
                    f'<div class="player-badge">{def_sel3}</div> '
                    f'<span style="color:#9CA3AF; font-size:0.9rem;">{" · ".join(meta_parts)}</span>',
                    unsafe_allow_html=True,
                )
                if def_player.avg_ppp_def:
                    st.caption(f"Avg PPP allowed: {def_player.avg_ppp_def:.3f}")

            # Charts
            ch1, ch2 = st.columns(2)
            with ch1:
                st.plotly_chart(
                    plot_similarity_scores(similar, def_sel3),
                    use_container_width=True,
                )

            if def_player:
                styled = plot_similarity_comparison(def_player, similar, graph, top_k=3)
                st.markdown("**Defensive Stat Comparison** — green = better than target, red = worse")
                st.dataframe(styled, use_container_width=True)

            st.markdown("---")
            st.markdown('<div class="section-header">Similarity Rankings</div>',
                        unsafe_allow_html=True)

            # Table
            sim_rows = []
            for s in similar:
                other = graph.players.get(s["defender_id"])
                sim_rows.append({
                    "Defender": s["defender"],
                    "Team": s.get("team") or "—",
                    "Position": s.get("position") or "—",
                    "Height": other.height if other else "—",
                    "Weight": f"{other.weight} lbs" if other and other.weight else "—",
                    "Combined Score": f"{s['combined_score']:.3f}",
                    "Jaccard": f"{s['jaccard']:.3f}",
                    "Cosine": f"{s['cosine']:.3f}",
                    "Correlation": f"{s['correlation']:.3f}",
                    "Shared Opp.": s["shared_opponents"],
                    "Avg PPP Allowed": f"{s['avg_ppp_def']:.3f}" if s.get("avg_ppp_def") else "—",
                })

            st.dataframe(pd.DataFrame(sim_rows), hide_index=True, use_container_width=True)

            # Methodology explanation
            with st.expander("How is similarity calculated?"):
                st.markdown(
                    """
                    **Graph-Based Defensive Similarity**

                    Two defenders are considered *similar* if they guard the same set of offensive
                    players and allow similar efficiency against each one.

                    **Combined Score = 0.4 × Jaccard + 0.3 × Cosine + 0.3 × (Correlation + 1) / 2**

                    | Component | Meaning |
                    |---|---|
                    | **Jaccard** | Fraction of shared offensive opponents (|A ∩ B| / |A ∪ B|) |
                    | **Cosine** | PPP pattern alignment over shared opponents |
                    | **Correlation** | Linear correlation of PPP values across shared opponents |

                    Only pairs with ≥ 3 shared opponents are considered.
                    """
                )


# ===========================================================================
# TAB 4 — LLM Scouting Report
# ===========================================================================
with tab4:
    st.markdown('<div class="section-header">AI Scouting Report</div>', unsafe_allow_html=True)
    st.markdown(
        "Generate natural-language scouting reports synthesizing matchup graph data. "
        "Powered by Claude (Anthropic)."
    )

    if not st.session_state.api_key:
        st.markdown(
            '<div class="info-box">⚠️ Enter your <b>Anthropic API key</b> in the sidebar to enable scouting reports.</div>',
            unsafe_allow_html=True,
        )

    report_type = st.radio(
        "Report Type",
        ["Matchup Report", "Player Profile Report", "Defensive Similarity Report", "Team Matchup Report"],
        horizontal=True,
    )

    if report_type == "Matchup Report":
        r1, r2 = st.columns(2)
        with r1:
            off_r = st.selectbox("Offensive Player", graph.all_player_names("offense"), key="lr_off")
        with r2:
            def_r = st.selectbox("Defensive Player", graph.all_player_names("defense"), key="lr_def")

        if st.button("Generate Matchup Report", type="primary", disabled=not st.session_state.api_key):
            edge = graph.get_matchup(off_r, def_r)
            off_pid = graph.find_player_id(off_r)
            def_pid = graph.find_player_id(def_r)
            off_p = graph.players.get(off_pid)
            def_p = graph.players.get(def_pid)

            if edge and off_p and def_p:
                with st.spinner("Generating scouting report…"):
                    report = generate_matchup_report(
                        edge, off_p, def_p,
                        graph.get_offensive_neighborhood(off_r, top_n=8),
                        graph.get_defensive_neighborhood(def_r, top_n=8),
                        st.session_state.api_key,
                    )
                st.markdown("---")
                st.markdown(f"### Scouting Report: {off_r} vs {def_r}")
                st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
                # CounterPoint flags — show for both players if drift detected
                off_pid_r = graph.find_player_id(off_r)
                def_pid_r = graph.find_player_id(def_r)
                if off_pid_r:
                    _render_cp_flag(off_pid_r, off_r)
                if def_pid_r:
                    _render_cp_flag(def_pid_r, def_r)
            else:
                st.warning("No direct matchup found for that pair with sufficient possessions.")

    elif report_type == "Player Profile Report":
        pp_r_player = st.selectbox("Player", graph.all_player_names(), key="lr_player")
        pp_r_role = st.radio("Role", ["offense", "defense"], horizontal=True, key="lr_role")

        if st.button("Generate Player Report", type="primary", disabled=not st.session_state.api_key):
            pid = graph.find_player_id(pp_r_player)
            player = graph.players.get(pid)
            if player:
                hood = (graph.get_offensive_neighborhood(pp_r_player, top_n=10)
                        if pp_r_role == "offense"
                        else graph.get_defensive_neighborhood(pp_r_player, top_n=10))
                with st.spinner("Generating scouting report…"):
                    report = generate_player_profile_report(
                        player, pp_r_role, hood, st.session_state.api_key
                    )
                st.markdown("---")
                st.markdown(f"### Scouting Report: {pp_r_player} ({pp_r_role.title()})")
                st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
                # CounterPoint flag
                pp_pid_r = graph.find_player_id(pp_r_player)
                if pp_pid_r:
                    _render_cp_flag(pp_pid_r, pp_r_player)
            else:
                st.warning("Player not found.")

    elif report_type == "Defensive Similarity Report":
        ds_r_def = st.selectbox("Defender", graph.all_player_names("defense"), key="lr_def2")

        if st.button("Generate Similarity Report", type="primary", disabled=not st.session_state.api_key):
            pid = graph.find_player_id(ds_r_def)
            player = graph.players.get(pid)
            similar = graph.find_similar_defenders(ds_r_def, top_n=6)
            if player and similar:
                with st.spinner("Generating scouting report…"):
                    report = generate_similarity_report(
                        player, similar, graph, st.session_state.api_key
                    )
                st.markdown("---")
                st.markdown(f"### Scouting Report: {ds_r_def} — Defensive Similarity")
                st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
            elif not similar:
                st.warning("Not enough data for similarity report.")
            else:
                st.warning("Player not found.")

    else:  # Team Matchup Report
        if not st.session_state.get("team_data_loaded"):
            st.markdown(
                '<div class="info-box">⚠️ Load team data first — click <b>Load Team Data</b> in the '
                '<b>Team Matchup</b> tab to enable this report.</div>',
                unsafe_allow_html=True,
            )
        else:
            _tdf_llm = st.session_state.get("team_stats_df", pd.DataFrame())
            _team_name_col_llm = next(
                (c for c in ["TEAM_NAME", "Team", "TeamName"] if c in _tdf_llm.columns), None
            )
            if _tdf_llm.empty or not _team_name_col_llm:
                st.warning("Team stats data not available.")
            else:
                _llm_team_names = sorted(_tdf_llm[_team_name_col_llm].dropna().unique().tolist())
                _tm_col1, _tm_col2 = st.columns(2)
                with _tm_col1:
                    _llm_t1_def = _llm_team_names.index("Los Angeles Lakers") if "Los Angeles Lakers" in _llm_team_names else 0
                    _llm_team1 = st.selectbox("Team 1", _llm_team_names, index=_llm_t1_def, key="llm_team1")
                with _tm_col2:
                    _llm_t2_def = _llm_team_names.index("Boston Celtics") if "Boston Celtics" in _llm_team_names else min(1, len(_llm_team_names) - 1)
                    _llm_team2 = st.selectbox("Team 2", _llm_team_names, index=_llm_t2_def, key="llm_team2")

                if st.button("Generate Team Matchup Report", type="primary", disabled=not st.session_state.api_key):
                    _llm_t1_row = _tdf_llm[_tdf_llm[_team_name_col_llm] == _llm_team1]
                    _llm_t2_row = _tdf_llm[_tdf_llm[_team_name_col_llm] == _llm_team2]
                    if _llm_t1_row.empty or _llm_t2_row.empty:
                        st.warning("Could not find stats for one or both teams.")
                    else:
                        _llm_t1_stats = _llm_t1_row.iloc[0].to_dict()
                        _llm_t2_stats = _llm_t2_row.iloc[0].to_dict()
                        with st.spinner("Generating team matchup report…"):
                            report = generate_team_matchup_report(
                                _llm_team1, _llm_team2,
                                _llm_t1_stats, _llm_t2_stats,
                                graph, st.session_state.api_key,
                            )
                        st.markdown("---")
                        st.markdown(f"### Scouting Report: {_llm_team1} vs {_llm_team2}")
                        st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)


# ===========================================================================
# TAB 5 — Graph Overview
# ===========================================================================
with tab5:
    st.markdown('<div class="section-header">Graph Overview</div>', unsafe_allow_html=True)

    summ = graph.get_summary()
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Players", summ["total_nodes"])
    m2.metric("Offensive", summ["offensive_players"])
    m3.metric("Defensive", summ["defensive_players"])
    m4.metric("Matchup Edges", summ["total_edges"])
    m5.metric("Graph Density", f"{summ['density']:.4f}")
    m6.metric("Avg PPP", f"{summ['avg_ppp']:.3f}")

    st.markdown("---")

    # Degree distribution
    off_degs, def_degs = graph.degree_sequences()
    st.plotly_chart(plot_degree_distribution(off_degs, def_degs), use_container_width=True)

    # PPP Heatmap
    st.markdown('<div class="section-header">PPP Heatmap</div>', unsafe_allow_html=True)
    st.markdown("Points per possession for the most active offensive vs. defensive players.")
    hmap_n = st.slider("Players per axis", 5, 20, 12, key="hmap_n")
    st.plotly_chart(plot_ppp_heatmap(graph, top_n=hmap_n), use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Most Connected Players</div>', unsafe_allow_html=True)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Top Offensive Players (most defenders faced)**")
        top_off = graph.top_connected("offense", top_n=15)
        st.dataframe(pd.DataFrame(top_off), hide_index=True, use_container_width=True)

    with t2:
        st.markdown("**Top Defensive Players (most scorers guarded)**")
        top_def = graph.top_connected("defense", top_n=15)
        st.dataframe(pd.DataFrame(top_def), hide_index=True, use_container_width=True)


# ===========================================================================
# TAB 7 — CounterPoint
# ===========================================================================
with tab7:
    st.markdown('<div class="section-header">CounterPoint Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        "Data-driven system that detects when conventional scouting wisdom is "
        "statistically outdated — quantifying the gap between a player's reputation "
        "and their current performance."
    )

    if not st.session_state.api_key:
        st.markdown(
            '<div class="info-box">⚠️ Enter your <b>Anthropic API key</b> in the sidebar '
            'to enable CounterPoint briefings and Q&amp;A.</div>',
            unsafe_allow_html=True,
        )

    # ── Team selectors ────────────────────────────────────────────────────────
    # Prefer team names from the loaded team stats; fall back to graph player teams.
    _cp_team_names: list = []
    if st.session_state.get("team_data_loaded") and st.session_state.get("team_stats_df") is not None:
        _cp_tnc = next(
            (c for c in ["TEAM_NAME", "TeamName"] if c in st.session_state.team_stats_df.columns),
            None,
        )
        if _cp_tnc:
            _cp_team_names = sorted(st.session_state.team_stats_df[_cp_tnc].dropna().tolist())

    if not _cp_team_names:
        _cp_team_names = sorted(set(p.team for p in graph.players.values() if p.team))

    if not _cp_team_names:
        st.markdown(
            '<div class="info-box">Load matchup data and enrich players (sidebar) to enable '
            'CounterPoint team analysis.</div>',
            unsafe_allow_html=True,
        )
    else:
        _cp_col1, _cp_col2 = st.columns(2)
        with _cp_col1:
            _def_t1 = (
                _cp_team_names.index("Los Angeles Lakers")
                if "Los Angeles Lakers" in _cp_team_names else 0
            )
            _cp_t1 = st.selectbox("Team 1", _cp_team_names, index=_def_t1, key="cp_team1_sel")
        with _cp_col2:
            _def_t2 = (
                _cp_team_names.index("Boston Celtics")
                if "Boston Celtics" in _cp_team_names
                else min(1, len(_cp_team_names) - 1)
            )
            _cp_t2 = st.selectbox("Team 2", _cp_team_names, index=_def_t2, key="cp_team2_sel")

        # Persist selected teams to session state so Q&A and flag navigation can read them
        st.session_state.cp_team1 = _cp_t1
        st.session_state.cp_team2 = _cp_t2

        # Team stats rows (for Claude context)
        _cp_t1_stats: dict | None = None
        _cp_t2_stats: dict | None = None
        if st.session_state.get("team_stats_df") is not None:
            _cp_tdf = st.session_state.team_stats_df
            _cp_tnc2 = next(
                (c for c in ["TEAM_NAME", "TeamName"] if c in _cp_tdf.columns), None
            )
            if _cp_tnc2:
                _r1 = _cp_tdf[_cp_tdf[_cp_tnc2] == _cp_t1]
                _r2 = _cp_tdf[_cp_tdf[_cp_tnc2] == _cp_t2]
                if not _r1.empty:
                    _cp_t1_stats = _r1.iloc[0].to_dict()
                if not _r2.empty:
                    _cp_t2_stats = _r2.iloc[0].to_dict()

        # ── Cross-team matchup edges ──────────────────────────────────────────
        _cp_matchups = get_cross_team_matchups(graph, _cp_t1, _cp_t2, top_n=12)

        # ── Analyse button ────────────────────────────────────────────────────
        _cp_analyse_btn = st.button(
            "⚡ Run CounterPoint Analysis",
            type="primary",
            key="cp_analyse",
        )

        if _cp_analyse_btn or st.session_state.cp_matchup_drift:
            if _cp_analyse_btn:
                # Compute drift for every offensive player in the cross-team matchups
                _cp_pids_to_run = list({m["off_pid"] for m in _cp_matchups})
                _cp_prog = st.progress(0, text="Computing narrative drift…")
                for _ci, _cpid in enumerate(_cp_pids_to_run):
                    _cp_prog.progress(
                        (_ci + 1) / max(len(_cp_pids_to_run), 1),
                        text=f"Analysing {graph.players[_cpid].name if _cpid in graph.players else _cpid}…",
                    )
                    if _cpid not in st.session_state.cp_matchup_drift:
                        try:
                            _cp_splits = get_player_career_splits(_cpid)
                            _cp_pname = graph.players[_cpid].name if _cpid in graph.players else ""
                            _cp_result = compute_drift(_cpid, _cp_splits, st.session_state.season, player_name=_cp_pname)
                        except Exception:
                            _cp_result = None
                        st.session_state.cp_matchup_drift[_cpid] = _cp_result
                _cp_prog.empty()

            # ──────────────────────────────────────────────────────────────────
            # SECTION 1 — Matchup Intelligence Panel
            # ──────────────────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown(
                '<div class="section-header">Matchup Intelligence Panel</div>',
                unsafe_allow_html=True,
            )

            # Collect flagged players for briefing + Q&A chips
            _cp_flagged: list = []

            if not _cp_matchups:
                st.info(
                    f"No cross-team matchup edges found between {_cp_t1} and {_cp_t2} "
                    f"in the loaded season data. Try a different team pair or lower the "
                    f"min possessions filter."
                )
            else:
                for _mi, _m in enumerate(_cp_matchups):
                    _off_pid  = _m["off_pid"]
                    _off_name = _m["off_player"]
                    _def_name = _m["def_player"]
                    _drift    = st.session_state.cp_matchup_drift.get(_off_pid)

                    if _drift and _drift.get("flagged"):
                        _flag   = _drift["flag"]
                        _color  = FLAG_COLOR.get(_flag, "#9CA3AF")
                        _flabel = FLAG_LABEL.get(_flag, "")
                        _stat   = _drift["max_drift_stat"]
                        _slbl   = CP_STAT_LABELS.get(_stat, _stat)

                        _cp_flagged.append({
                            "name":     _off_name,
                            "off_team": _m["off_team"],
                            "drift":    _drift,
                        })

                        st.markdown(
                            f'<div class="cp-entry" style="background:#1A2035; '
                            f'border-left: 4px solid {_color};">'
                            f'<div class="cp-player-name">{_off_name} '
                            f'<span style="color:#9CA3AF; font-size:0.82rem; font-weight:400;">'
                            f'({_m["off_team"]}) vs {_def_name} ({_m["def_team"]})</span></div>'
                            f'<div style="display:inline-block; background:{_color}33; '
                            f'color:{_color}; border:1px solid {_color}; border-radius:4px; '
                            f'padding:1px 8px; font-size:0.75rem; font-weight:700; '
                            f'margin-bottom:6px;">{_flabel}</div>'
                            f'<div class="cp-narrative"><b>The narrative:</b> '
                            f'{_drift["narrative"]}</div>'
                            f'<div class="cp-numbers" style="color:{_color};">'
                            f'<b>The numbers say:</b> {_drift["numbers_say"]}</div>'
                            f'<div class="cp-coaching"><b>Coaching implication:</b> '
                            f'{_drift["coaching_impl"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        # Sparkline for the flagged stat
                        _traj = _drift["trajectories"].get(_stat, {})
                        if _traj.get("seasons") and len(_traj["seasons"]) >= 2:
                            _spark_key = f"sparkline_{_off_name.replace(' ', '_')}_{_stat}_{_mi}"
                            st.plotly_chart(
                                plot_sparkline(
                                    _traj["seasons"],
                                    _traj["values"],
                                    _slbl,
                                    _flag,
                                ),
                                use_container_width=False,
                                config={"displayModeBar": False},
                                key=_spark_key,
                            )
                    elif _drift and not _drift.get("flagged"):
                        # Stable player — show actual stat summary instead of generic message
                        _stable_txt = _drift.get("stable_summary", "")
                        st.markdown(
                            f'<div class="cp-entry" style="background:#1A2035; '
                            f'border-left: 4px solid #2A3550;">'
                            f'<div class="cp-player-name">{_off_name} '
                            f'<span style="color:#9CA3AF; font-size:0.82rem; font-weight:400;">'
                            f'({_m["off_team"]}) vs {_def_name} ({_m["def_team"]})</span></div>'
                            f'<div style="color:#6B7280; font-size:0.82rem;">'
                            f'{_stable_txt if _stable_txt else "Scouting report is holding up — no significant narrative drift detected."}'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        # No drift data computed yet
                        st.markdown(
                            f'<div class="cp-entry" style="background:#1A2035; '
                            f'border-left: 4px solid #2A3550;">'
                            f'<div class="cp-player-name">{_off_name} '
                            f'<span style="color:#9CA3AF; font-size:0.82rem; font-weight:400;">'
                            f'({_m["off_team"]}) vs {_def_name} ({_m["def_team"]})</span></div>'
                            f'<div style="color:#6B7280; font-size:0.82rem;">'
                            f'No narrative drift detected — conventional scouting report is holding up.</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # ── Claude briefing ───────────────────────────────────────────
                if _cp_flagged and st.session_state.api_key:
                    st.markdown("<br>", unsafe_allow_html=True)
                    _cp_brief_btn = st.button(
                        "Generate Pre-Series Intelligence Briefing",
                        key="cp_brief_btn",
                        type="primary",
                    )
                    if _cp_brief_btn:
                        with st.spinner("CounterPoint is writing the briefing…"):
                            st.session_state.cp_briefing = call_cp_briefing(
                                _cp_flagged, _cp_t1, _cp_t2,
                                _cp_t1_stats, _cp_t2_stats,
                                st.session_state.api_key,
                            )
                    if st.session_state.cp_briefing:
                        st.markdown(
                            f'<div class="cp-briefing">{st.session_state.cp_briefing}</div>',
                            unsafe_allow_html=True,
                        )
                elif not st.session_state.api_key:
                    st.caption("Add an API key in the sidebar to generate the pre-series briefing.")

            # ──────────────────────────────────────────────────────────────────
            # SECTION 2 — Most Misread Players Leaderboard
            # ──────────────────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown(
                '<div class="section-header">Most Misread Players This Postseason</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                "Players with the largest gap between their established reputation and current "
                "season reality — ranked by narrative drift score."
            )

            _lb_col1, _lb_col2 = st.columns([2, 1])
            with _lb_col2:
                _lb_compute_btn = st.button(
                    "Compute Leaderboard",
                    key="cp_lb_btn",
                    help="Fetches career splits for all active players. May take 1-2 minutes.",
                )

            if _lb_compute_btn:
                # Compute drift for the top 80 players by total possessions
                _lb_players_sorted = sorted(
                    graph.players.values(),
                    key=lambda p: (p.off_matchup_count or 0) + (p.def_matchup_count or 0),
                    reverse=True,
                )[:80]
                _lb_prog = st.progress(0, text="Building leaderboard…")
                for _li, _lp in enumerate(_lb_players_sorted):
                    _lb_prog.progress(
                        (_li + 1) / len(_lb_players_sorted),
                        text=f"Analysing {_lp.name}…",
                    )
                    if _lp.player_id not in st.session_state.cp_leaderboard_drift:
                        try:
                            _lb_splits = get_player_career_splits(_lp.player_id)
                            _lb_result = compute_drift(
                                _lp.player_id, _lb_splits, st.session_state.season,
                                player_name=_lp.name,
                            )
                        except Exception:
                            _lb_result = None
                        st.session_state.cp_leaderboard_drift[_lp.player_id] = _lb_result
                _lb_prog.empty()

            # Render leaderboard if data exists
            _lb_data = {
                pid: d for pid, d in st.session_state.cp_leaderboard_drift.items()
                if d is not None and d.get("flagged")
            }

            if not _lb_data:
                if not _lb_compute_btn:
                    st.markdown(
                        '<div class="info-box">Click <b>Compute Leaderboard</b> above to rank '
                        'players by narrative drift score. Analyses the top 80 most-active '
                        'players in the loaded matchup data.</div>',
                        unsafe_allow_html=True,
                    )
            else:
                # Sort all flagged players by absolute drift score
                _lb_ranked = sorted(
                    _lb_data.items(),
                    key=lambda x: abs(x[1]["max_drift_score"]),
                    reverse=True,
                )

                # Try to split by conference using standings data
                _lb_conf_map: Dict[int, str] = {}
                _sdf_lb = st.session_state.get("standings_df")
                if _sdf_lb is not None and not _sdf_lb.empty:
                    _conf_col_lb = next(
                        (c for c in ["Conference", "TeamConference"] if c in _sdf_lb.columns),
                        None,
                    )
                    _nm_col_lb = next(
                        (c for c in ["FULL_NAME", "TeamName"] if c in _sdf_lb.columns), None
                    )
                    if _conf_col_lb and _nm_col_lb:
                        for _, _srow in _sdf_lb.iterrows():
                            _conf_val = str(_srow.get(_conf_col_lb, "")).upper()
                            _conf_str = "East" if _conf_val.startswith("E") else "West"
                            _tnm = str(_srow.get(_nm_col_lb, "")).lower()
                            for _pid, _d in _lb_ranked:
                                _pl = graph.players.get(_pid)
                                if _pl and _pl.team and _pl.team.lower() in _tnm:
                                    _lb_conf_map[_pid] = _conf_str

                _show_conferences = bool(_lb_conf_map)

                def _render_lb_entries(entries, label):
                    if label:
                        st.markdown(f"**{label} Conference**")
                    for rank, (pid, d) in enumerate(entries[:5], start=1):
                        _pl = graph.players.get(pid)
                        if not _pl:
                            continue
                        _fl    = FLAG_LABEL.get(d["flag"], "")
                        _col   = FLAG_COLOR.get(d["flag"], "#9CA3AF")
                        _sl    = CP_STAT_LABELS.get(d["max_drift_stat"], d["max_drift_stat"])
                        _ca    = d["career_avgs"].get(d["max_drift_stat"])
                        _cv    = d["current_vals"].get(d["max_drift_stat"])
                        _fmt   = ".1%" if "pct" in d["max_drift_stat"] or d["max_drift_stat"] == "ft_rate" else ".1f"
                        _ca_s  = f"{_ca:{_fmt}}" if _ca is not None else "—"
                        _cv_s  = f"{_cv:{_fmt}}" if _cv is not None else "—"
                        _z     = d["max_drift_score"]
                        st.markdown(
                            f'<div class="cp-leaderboard-row">'
                            f'<span style="color:#9CA3AF; font-size:0.82rem;">#{rank}</span>&nbsp;&nbsp;'
                            f'<b style="color:#FAFAFA;">{_pl.name}</b>&nbsp;'
                            f'<span style="color:#9CA3AF; font-size:0.82rem;">{_pl.team or ""}</span>'
                            f'&nbsp;&nbsp;'
                            f'<span style="background:{_col}33; color:{_col}; border:1px solid {_col}; '
                            f'border-radius:4px; padding:1px 6px; font-size:0.75rem;">{_fl}</span>'
                            f'<br>'
                            f'<span style="color:#9CA3AF; font-size:0.82rem;">Driving stat: '
                            f'<b style="color:#D1D5DB;">{_sl}</b> — career {_ca_s} → this season {_cv_s} '
                            f'({_z:+.1f}\u03c3)</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                if _show_conferences:
                    _east_entries = [(pid, d) for pid, d in _lb_ranked if _lb_conf_map.get(pid) == "East"]
                    _west_entries = [(pid, d) for pid, d in _lb_ranked if _lb_conf_map.get(pid) == "West"]
                    _lb_c1, _lb_c2 = st.columns(2)
                    with _lb_c1:
                        _render_lb_entries(_east_entries, "Eastern")
                    with _lb_c2:
                        _render_lb_entries(_west_entries, "Western")
                else:
                    _render_lb_entries(_lb_ranked, "")

            # ──────────────────────────────────────────────────────────────────
            # SECTION 3 — Ask CounterPoint
            # ──────────────────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown(
                '<div class="section-header">Ask CounterPoint</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                "Query the dashboard's own computed data in plain English. "
                "CounterPoint answers using only the drift scores and stats loaded here — "
                "not general basketball knowledge."
            )

            if not st.session_state.api_key:
                st.markdown(
                    '<div class="info-box">Add an Anthropic API key in the sidebar to enable Ask CounterPoint.</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Combine matchup drift + leaderboard drift for Q&A context
                _qa_state = {
                    **st.session_state.cp_leaderboard_drift,
                    **st.session_state.cp_matchup_drift,
                }

                # Example question chips (dynamic based on flagged players)
                _example_qs = generate_example_questions(_cp_t1, _cp_t2, _cp_flagged)

                # Show conversation history
                for _msg in st.session_state.cp_chat_history:
                    _role_label = "You" if _msg["role"] == "user" else "CounterPoint"
                    _role_color = "#9CA3AF" if _msg["role"] == "user" else "#F0A500"
                    st.markdown(
                        f'<div class="cp-response-card">'
                        f'<div class="cp-response-header" style="color:{_role_color};">'
                        f'{_role_label}</div>'
                        f'<div style="color:#E5E7EB; font-size:0.93rem; line-height:1.65; '
                        f'white-space:pre-wrap;">{_msg["content"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Input form
                with st.form("cp_qa_form", clear_on_submit=True):
                    _cp_query = st.text_input(
                        "Ask a question",
                        placeholder="e.g. Who is the most misread player in this series?",
                        label_visibility="collapsed",
                    )
                    _cp_submit = st.form_submit_button("Ask CounterPoint", type="primary")

                # Suggestion chips — clicking any populates a follow-up question
                _chip_cols = st.columns(3)
                _chip_clicked: str = ""
                for _ci, (_chip_col, _q) in enumerate(zip(_chip_cols, _example_qs)):
                    with _chip_col:
                        if st.button(
                            _q,
                            key=f"cp_chip_{_ci}",
                            use_container_width=True,
                        ):
                            _chip_clicked = _q

                # Handle form submission or chip click
                _cp_user_input = _cp_query if _cp_submit and _cp_query.strip() else _chip_clicked
                if _cp_user_input:
                    with st.spinner("CounterPoint is thinking…"):
                        _cp_answer = call_cp_qa(
                            _cp_user_input,
                            graph,
                            _qa_state,
                            _cp_t1,
                            _cp_t2,
                            _cp_t1_stats,
                            _cp_t2_stats,
                            st.session_state.cp_chat_history,
                            st.session_state.api_key,
                        )
                    # Update conversation history (keep last 3 exchanges = 6 messages)
                    st.session_state.cp_chat_history.append(
                        {"role": "user", "content": _cp_user_input}
                    )
                    st.session_state.cp_chat_history.append(
                        {"role": "assistant", "content": _cp_answer}
                    )
                    st.session_state.cp_chat_history = st.session_state.cp_chat_history[-6:]
                    st.rerun()

                # Clear conversation button
                if st.session_state.cp_chat_history:
                    if st.button("Clear conversation", key="cp_clear"):
                        st.session_state.cp_chat_history = []
                        st.rerun()
# ===========================================================================
# TAB 6 — Team Matchup
# ===========================================================================
with tab6:
    import math

    st.markdown('<div class="section-header">Team Matchup</div>', unsafe_allow_html=True)
    st.markdown("Compare any two NBA teams head-to-head using advanced team stats and standings.")

    # ---- Load / Refresh buttons ----
    _btn_col1, _btn_col2, _updated_col = st.columns([1, 1, 4])
    with _btn_col1:
        load_team_btn = st.button(
            "📥 Load Team Data", type="primary",
            help="Fetch live team stats, standings, and playoff bracket from NBA.com",
        )
    with _btn_col2:
        refresh_team_btn = st.button(
            "🔄 Refresh",
            help="Clear cache and re-fetch all live team data",
            disabled=not st.session_state.team_data_loaded,
        )
    with _updated_col:
        if st.session_state.get("team_data_updated_at"):
            st.caption(f"Last updated: {st.session_state.team_data_updated_at}")

    def _load_all_team_data(force: bool = False):
        import datetime
        _season = st.session_state.season
        _errors = []

        tdf = get_team_stats_live(season=_season, force_refresh=force)
        if tdf.empty:
            _errors.append("team stats")

        sdf = get_team_standings_live(season=_season, force_refresh=force)
        if sdf.empty:
            _errors.append("standings")

        psdf = get_playoff_series(season=_season, force_refresh=force)

        # Build team_id lookup from standings
        _tid_map = {}
        if not sdf.empty:
            _id_col = next((c for c in ["TeamID", "TEAM_ID"] if c in sdf.columns), None)
            _nm_col = next((c for c in ["FULL_NAME", "TeamName", "TEAM_NAME"] if c in sdf.columns), None)
            if _id_col and _nm_col:
                for _, _r in sdf.iterrows():
                    _tid_map[str(_r[_nm_col])] = int(_r[_id_col])

        st.session_state.team_stats_df = tdf if not tdf.empty else None
        st.session_state.standings_df = sdf if not sdf.empty else None
        st.session_state.playoff_series_df = psdf if not psdf.empty else None
        st.session_state.team_data_loaded = True
        st.session_state.roster_team_ids = _tid_map
        st.session_state.team_data_updated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if force:
            # Clear roster cache so re-fetch picks up fresh data
            st.session_state.roster_cache = {}

        return _errors

    if load_team_btn:
        with st.spinner("Fetching live team stats, standings, and playoff bracket from NBA.com…"):
            _errs = _load_all_team_data(force=False)
        if st.session_state.team_stats_df is not None:
            st.success(f"Loaded stats for {len(st.session_state.team_stats_df)} teams.")
        if _errs:
            st.warning(f"Some endpoints unavailable: {', '.join(_errs)}. Displaying last known values where possible.")

    if refresh_team_btn:
        with st.spinner("Clearing cache and re-fetching live data…"):
            _errs = _load_all_team_data(force=True)
        if st.session_state.team_stats_df is not None:
            st.success("Data refreshed.")
        if _errs:
            st.warning(f"Live data temporarily unavailable for: {', '.join(_errs)}. Displaying last known values.")

    if not st.session_state.team_data_loaded or st.session_state.team_stats_df is None:
        st.markdown(
            '<div class="info-box">Click <b>Load Team Data</b> above to fetch NBA team stats '
            'and standings for the selected season.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    _tdf = st.session_state.team_stats_df
    _sdf = st.session_state.standings_df

    # Resolve team name column — live API uses TEAM_NAME
    _team_name_col = "TEAM_NAME" if "TEAM_NAME" in _tdf.columns else _tdf.columns[1]
    _team_names = sorted(_tdf[_team_name_col].dropna().tolist())

    # ---- Team selectors ----
    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        default_t1 = _team_names.index("Los Angeles Lakers") if "Los Angeles Lakers" in _team_names else 0
        team1_sel = st.selectbox("Team 1", _team_names, index=default_t1, key="tm_team1")
    with sel_col2:
        default_t2 = _team_names.index("Boston Celtics") if "Boston Celtics" in _team_names else min(1, len(_team_names) - 1)
        team2_sel = st.selectbox("Team 2", _team_names, index=default_t2, key="tm_team2")

    # Ensure rosters are loaded for the two selected teams (cached per team_id)
    def _ensure_roster(team_name: str) -> pd.DataFrame:
        _tid = st.session_state.roster_team_ids.get(team_name)
        if not _tid:
            # Try fuzzy match from standings
            _sdf_r = st.session_state.standings_df
            if _sdf_r is not None and not _sdf_r.empty:
                _id_col = next((c for c in ["TeamID", "TEAM_ID"] if c in _sdf_r.columns), None)
                _nm_cols = [c for c in ["FULL_NAME", "TeamName", "TeamCity", "TEAM_NAME"] if c in _sdf_r.columns]
                for _nm_col in _nm_cols:
                    _match = _sdf_r[_sdf_r[_nm_col].astype(str).str.lower().str.contains(
                        team_name.split()[-1].lower(), na=False
                    )]
                    if not _match.empty and _id_col:
                        _tid = int(_match.iloc[0][_id_col])
                        st.session_state.roster_team_ids[team_name] = _tid
                        break
        if not _tid:
            return pd.DataFrame()
        if _tid not in st.session_state.roster_cache:
            with st.spinner(f"Fetching {team_name} roster…"):
                st.session_state.roster_cache[_tid] = get_team_roster(
                    _tid, season=st.session_state.season
                )
        return st.session_state.roster_cache.get(_tid, pd.DataFrame())

    _roster_t1 = _ensure_roster(team1_sel)
    _roster_t2 = _ensure_roster(team2_sel)

    def _roster_player_names(roster_df: pd.DataFrame) -> list:
        """Extract player names from a roster DataFrame."""
        for col in ["PLAYER", "PlayerName", "PLAYER_NAME", "Name"]:
            if col in roster_df.columns:
                return sorted(roster_df[col].dropna().tolist())
        return []

    # Fetch rows as dicts
    _t1_row = _tdf[_tdf[_team_name_col] == team1_sel]
    _t2_row = _tdf[_tdf[_team_name_col] == team2_sel]

    if _t1_row.empty or _t2_row.empty:
        st.warning("Could not find stats for one or both selected teams.")
        st.stop()

    _t1 = _t1_row.iloc[0].to_dict()
    _t2 = _t2_row.iloc[0].to_dict()

    st.markdown("---")

    # ===========================================================
    # Section 1: Head-to-Head Comparison
    # ===========================================================
    st.markdown('<div class="section-header">Head-to-Head Comparison</div>', unsafe_allow_html=True)

    # Key stat metrics row
    _stat_meta = [
        ("Net Rtg",  "NET_RATING",  False),
        ("Off Rtg",  "OFF_RATING",  False),
        ("Def Rtg",  "DEF_RATING",  True),   # lower is better → inverse delta
        ("eFG%",     "EFG_PCT",     False),
        ("TOV%",     "TM_TOV_PCT",  True),
        ("Pace",     "PACE",        False),
    ]

    _mcols = st.columns(len(_stat_meta))
    for _col, (label, key, invert) in zip(_mcols, _stat_meta):
        _v1 = _t1.get(key)
        _v2 = _t2.get(key)
        if _v1 is not None and _v2 is not None:
            _v1f, _v2f = float(_v1), float(_v2)
            _delta = _v1f - _v2f
            _is_pct = key in ("EFG_PCT", "TM_TOV_PCT", "OREB_PCT", "TS_PCT")
            _val_str = f"{_v1f:.1%}" if _is_pct else f"{_v1f:.1f}"
            _dlt_str = f"{_delta:+.1%}" if _is_pct else f"{_delta:+.1f}"
            _dlt_color = "inverse" if invert else "normal"
            _col.metric(f"{label} ({team1_sel.split()[-1]})", _val_str,
                        delta=f"{_dlt_str} vs {team2_sel.split()[-1]}",
                        delta_color=_dlt_color)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts side by side
    _chart_col1, _chart_col2 = st.columns([1.1, 1])
    with _chart_col1:
        st.plotly_chart(
            plot_team_radar(_t1, _t2, team1_sel, team2_sel),
            use_container_width=True,
        )
    with _chart_col2:
        st.plotly_chart(
            plot_team_comparison_bars(_t1, _t2, team1_sel, team2_sel),
            use_container_width=True,
        )

    # Advantage summary table
    st.markdown('<div class="section-header">Advantage Summary</div>', unsafe_allow_html=True)

    _adv_rows = []
    _adv_stats = [
        ("Off Rating",  "OFF_RATING",  False, False),
        ("Def Rating",  "DEF_RATING",  True,  False),
        ("Net Rating",  "NET_RATING",  False, False),
        ("Pace",        "PACE",        False, False),
        ("eFG%",        "EFG_PCT",     False, True),
        ("TOV%",        "TM_TOV_PCT",  True,  True),
        ("OReb%",       "OREB_PCT",    False, True),
        ("TS%",         "TS_PCT",      False, True),
    ]
    for _lbl, _key, _lower_better, _as_pct in _adv_stats:
        _v1 = _t1.get(_key)
        _v2 = _t2.get(_key)
        if _v1 is None or _v2 is None:
            continue
        _v1f, _v2f = float(_v1), float(_v2)
        _fmt = (lambda x: f"{x:.1%}") if _as_pct else (lambda x: f"{x:.1f}")
        _edge = team2_sel if (_lower_better and _v1f > _v2f) or (not _lower_better and _v1f < _v2f) else team1_sel
        if abs(_v1f - _v2f) < 0.001:
            _edge = "Even"
        _adv_rows.append({
            "Stat": _lbl,
            team1_sel: _fmt(_v1f),
            team2_sel: _fmt(_v2f),
            "Advantage": _edge,
        })

    if _adv_rows:
        _adv_df = pd.DataFrame(_adv_rows)
        # Color the advantage column with HTML
        def _color_adv(val):
            if val == team1_sel:
                return f'<span style="color:{NAVY_CSS}; font-weight:700;">{val}</span>'
            elif val == team2_sel:
                return f'<span style="color:#C8102E; font-weight:700;">{val}</span>'
            return f'<span style="color:#6B7280;">{val}</span>'

        NAVY_CSS = "#4A90D9"  # slightly lighter navy for readability on dark bg
        _html_rows = ""
        for _, _r in _adv_df.iterrows():
            _adv_cell = _color_adv(_r["Advantage"])
            _html_rows += (
                f"<tr>"
                f"<td style='padding:6px 12px; color:#D1D5DB;'>{_r['Stat']}</td>"
                f"<td style='padding:6px 12px; color:#FAFAFA; text-align:center;'>{_r[team1_sel]}</td>"
                f"<td style='padding:6px 12px; color:#FAFAFA; text-align:center;'>{_r[team2_sel]}</td>"
                f"<td style='padding:6px 12px; text-align:center;'>{_adv_cell}</td>"
                f"</tr>"
            )

        _adv_html = f"""
        <table style='width:100%; border-collapse:collapse; background:#1A2035; border-radius:8px; overflow:hidden;'>
          <thead>
            <tr style='background:#0E1117;'>
              <th style='padding:8px 12px; color:#F0A500; text-align:left;'>Stat</th>
              <th style='padding:8px 12px; color:#4A90D9; text-align:center;'>{team1_sel}</th>
              <th style='padding:8px 12px; color:#C8102E; text-align:center;'>{team2_sel}</th>
              <th style='padding:8px 12px; color:#9CA3AF; text-align:center;'>Advantage</th>
            </tr>
          </thead>
          <tbody>{_html_rows}</tbody>
        </table>
        """
        st.markdown(_adv_html, unsafe_allow_html=True)

    st.markdown("---")

    # ===========================================================
    # Section 2: Playoff Predictor
    # ===========================================================
    st.markdown('<div class="section-header">Playoff Predictor</div>', unsafe_allow_html=True)

    if _sdf is None or _sdf.empty:
        st.markdown(
            '<div class="info-box">Standings data unavailable. Playoff projections cannot be shown.</div>',
            unsafe_allow_html=True,
        )
    else:
        # Determine column names flexibly
        _conf_col = next((c for c in ["Conference", "ConferenceAbbrev", "TeamConference"] if c in _sdf.columns), None)
        _name_col = next((c for c in ["TeamName", "Team", "TEAM_NAME"] if c in _sdf.columns), None)
        _wins_col = next((c for c in ["WINS", "WinPct", "Win"] if c in _sdf.columns), None)
        _losses_col = next((c for c in ["LOSSES", "Loss"] if c in _sdf.columns), None)
        _rank_col = next((c for c in ["PlayoffRank", "ConferenceRank", "Rank"] if c in _sdf.columns), None)

        # Use FULL_NAME (city + nickname) if available, else fall back to _name_col
        _full_name_col = "FULL_NAME" if "FULL_NAME" in _sdf.columns else _name_col

        # Detect play-in teams from series vector: if no R1 series exist yet,
        # seeds 7-10 are still competing in play-in games
        _psdf = st.session_state.get("playoff_series_df")
        _playin_active = True  # assume play-in pending until series vector says otherwise
        _confirmed_r1_teams: set = set()  # team full names confirmed in R1
        if _psdf is not None and not _psdf.empty:
            _round_col = next((c for c in ["ROUND", "SeriesRound", "SERIES_ROUND"] if c in _psdf.columns), None)
            if _round_col:
                _r1 = _psdf[_psdf[_round_col].astype(str) == "1"]
                if not _r1.empty:
                    _playin_active = False
                    # Collect team names from confirmed R1 series
                    for _tcol in ["HOME_TEAM_NAME", "VISITOR_TEAM_NAME", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]:
                        if _tcol in _r1.columns:
                            _confirmed_r1_teams.update(_r1[_tcol].dropna().astype(str).tolist())

        # Clinch / play-in indicator column from leaguestandingsv3
        _clinch_col = next((c for c in ["ClinchIndicator", "CLINCH_INDICATOR", "ClinchedIndicator"] if c in _sdf.columns), None)

        if _conf_col and _name_col:
            # Build seedings — use full sdf so we get clinch indicators
            _standings_full = _sdf.copy()

            def _get_conf_seeds(conf_label):
                _sub = _standings_full[_standings_full[_conf_col].str.upper().str.startswith(conf_label.upper())]
                if _rank_col and _rank_col in _sub.columns:
                    _sub = _sub.sort_values(_rank_col)
                elif _wins_col and _wins_col in _sub.columns:
                    _sub = _sub.sort_values(_wins_col, ascending=False)
                return _sub.head(10).reset_index(drop=True)  # include seeds 9-10 for play-in

            _east = _get_conf_seeds("E")
            _west = _get_conf_seeds("W")

            def _seed_label(sr, seed_num) -> str:
                """Build a seed line with play-in badge and win-loss."""
                _nm = sr.get(_full_name_col) or sr.get(_name_col, "—")
                _sw = sr.get(_wins_col, None)
                _sl = sr.get(_losses_col, None)
                try:
                    _wl = f" {int(float(_sw))}-{int(float(_sl))}" if _sw is not None and _sl is not None else ""
                except Exception:
                    _wl = ""
                _badge = ""
                if seed_num >= 7:
                    if _playin_active:
                        _badge = " <span style='font-size:0.75em;color:#F0A500;'>[Play-In]</span>"
                    elif _clinch_col and sr.get(_clinch_col) in ("pi", "PI"):
                        _badge = " <span style='font-size:0.75em;color:#F0A500;'>[Play-In]</span>"
                return _nm, _wl, _badge

            # Show seeds
            _seed_c1, _seed_c2 = st.columns(2)
            with _seed_c1:
                st.markdown("**Eastern Conference Seeds**")
                for _si, _sr in _east.iterrows():
                    _seed_num = _si + 1
                    _snm, _wl, _badge = _seed_label(_sr, _seed_num)
                    _is_sel = _snm in (team1_sel, team2_sel) or _sr.get(_name_col, "") in (team1_sel, team2_sel)
                    _style = "color:#F0A500; font-weight:700;" if _is_sel else ""
                    _sep = "──────────────" if _seed_num == 6 else ""
                    if _sep:
                        st.markdown(f"<span style='color:#2A3550; font-size:0.7em;'>{_sep}</span>", unsafe_allow_html=True)
                    st.markdown(f"<span style='{_style}'>{_seed_num}. {_snm}{_wl}</span>{_badge}", unsafe_allow_html=True)
            with _seed_c2:
                st.markdown("**Western Conference Seeds**")
                for _si, _sr in _west.iterrows():
                    _seed_num = _si + 1
                    _snm, _wl, _badge = _seed_label(_sr, _seed_num)
                    _is_sel = _snm in (team1_sel, team2_sel) or _sr.get(_name_col, "") in (team1_sel, team2_sel)
                    _style = "color:#F0A500; font-weight:700;" if _is_sel else ""
                    _sep = "──────────────" if _seed_num == 6 else ""
                    if _sep:
                        st.markdown(f"<span style='color:#2A3550; font-size:0.7em;'>{_sep}</span>", unsafe_allow_html=True)
                    st.markdown(f"<span style='{_style}'>{_seed_num}. {_snm}{_wl}</span>{_badge}", unsafe_allow_html=True)

            # -------------------------------------------------------
            # Injury adjustments
            # -------------------------------------------------------
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Injury Adjustments (optional)**")

            # Helper: resolve standings nickname → full name in team stats df
            def _resolve_team_row(nickname: str) -> pd.DataFrame:
                """Find team row in _tdf matching a nickname or full name (case-insensitive)."""
                # Exact match first
                _exact = _tdf[_tdf[_team_name_col].str.lower() == nickname.lower()]
                if not _exact.empty:
                    return _exact
                # Partial: nickname is contained in full name (e.g. "Thunder" in "Oklahoma City Thunder")
                _partial = _tdf[_tdf[_team_name_col].str.lower().str.contains(nickname.lower(), regex=False)]
                if not _partial.empty:
                    return _partial
                # Reverse: last word of full name matches nickname
                _last = _tdf[_tdf[_team_name_col].apply(lambda x: x.split()[-1].lower()) == nickname.lower()]
                return _last

            # Roster-based player lists (live from API, already fetched above)
            _t1_players = _roster_player_names(_roster_t1)
            _t2_players = _roster_player_names(_roster_t2)

            _inj_c1, _inj_c2 = st.columns(2)
            with _inj_c1:
                _t1_injured = st.multiselect(
                    f"{team1_sel} — Out",
                    _t1_players,
                    key="inj_t1",
                )
            with _inj_c2:
                _t2_injured = st.multiselect(
                    f"{team2_sel} — Out",
                    _t2_players,
                    key="inj_t2",
                )
            if not _t1_players and not _t2_players:
                st.caption(
                    "⚠️ Roster data unavailable — click **Refresh** or try a different season."
                )
            else:
                st.caption(
                    "Mark players as out. Their estimated impact (PIE × 100) is "
                    "subtracted from the team's effective Net Rating before computing probabilities."
                )

            def _injury_impact(injured_names) -> float:
                """Sum up PIE-based impact for injured players (PIE × 100 ≈ net rating pts)."""
                total = 0.0
                for nm in injured_names:
                    _pid = graph.find_player_id(nm)
                    p = graph.players.get(_pid) if _pid else None
                    if p and p.pie is not None:
                        total += float(p.pie) * 100
                return total

            _t1_impact = _injury_impact(_t1_injured)
            _t2_impact = _injury_impact(_t2_injured)

            # -------------------------------------------------------
            # Series win probability helper
            # -------------------------------------------------------
            def _series_prob(p: float) -> float:
                """Best-of-7 series win probability given single-game win prob p."""
                q = 1 - p
                return p**4 * (1 + 4*q + 10*q**2 + 20*q**3)

            # -------------------------------------------------------
            # Win probability: selected teams head-to-head
            # -------------------------------------------------------
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Series Win Probability: {team1_sel} vs {team2_sel}**")

            _net1 = _t1.get("NET_RATING")
            _net2 = _t2.get("NET_RATING")

            if _net1 is not None and _net2 is not None:
                _net1f = float(_net1) - _t1_impact
                _net2f = float(_net2) - _t2_impact
                _diff = _net1f - _net2f
                _p1g = 1 / (1 + math.exp(-(_diff / 7)))
                _p2g = 1 - _p1g
                _p1s = _series_prob(_p1g)
                _p2s = 1 - _p1s

                if _t1_injured or _t2_injured:
                    st.caption(
                        f"Adjusted Net Ratings — {team1_sel}: {_net1f:+.1f} "
                        f"(raw {float(_t1.get('NET_RATING')):+.1f}, −{_t1_impact:.1f} inj), "
                        f"{team2_sel}: {_net2f:+.1f} "
                        f"(raw {float(_t2.get('NET_RATING')):+.1f}, −{_t2_impact:.1f} inj)"
                    )

                _wp_col1, _wp_col2 = st.columns(2)
                with _wp_col1:
                    st.markdown(
                        f'<div class="stat-card"><h4>{team1_sel}</h4>'
                        f'<p>Series win prob: <span class="value">{_p1s:.1%}</span></p>'
                        f'<p>Per-game win prob: <span class="value">{_p1g:.1%}</span></p>'
                        f'<p>Adj Net Rating: <span class="value">{_net1f:+.1f}</span></p></div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(_p1s)
                with _wp_col2:
                    st.markdown(
                        f'<div class="stat-card"><h4>{team2_sel}</h4>'
                        f'<p>Series win prob: <span class="value">{_p2s:.1%}</span></p>'
                        f'<p>Per-game win prob: <span class="value">{_p2g:.1%}</span></p>'
                        f'<p>Adj Net Rating: <span class="value">{_net2f:+.1f}</span></p></div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(_p2s)

                st.caption(
                    "Series probability = P(win 4 of 7) using per-game win prob derived from "
                    "Net Rating differential: p = 1/(1+e^(−Δ/7)). Injury adjustment subtracts "
                    "PIE×100 from the team's Net Rating per injured player."
                )

            # -------------------------------------------------------
            # First-round matchup projections — expandable cards
            # -------------------------------------------------------
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">Projected First-Round Matchups</div>', unsafe_allow_html=True)

            def _predicted_series(fav: str, prob: float) -> str:
                """Return 'Fav in N' prediction string based on series win probability."""
                if prob >= 0.97:
                    return f"{fav} in 4"
                elif prob >= 0.88:
                    return f"{fav} in 5"
                elif prob >= 0.76:
                    return f"{fav} in 6"
                elif prob >= 0.60:
                    return f"{fav} in 7"
                else:
                    return "Toss-up"

            def _show_matchups(conf_seeds: pd.DataFrame, conf_label: str):
                st.markdown(f"**{conf_label}**")
                _pairs = [(1, 8), (2, 7), (3, 6), (4, 5)]
                for _high, _low in _pairs:
                    _hi_idx = _high - 1
                    _lo_idx = _low - 1
                    if _hi_idx >= len(conf_seeds) or _lo_idx >= len(conf_seeds):
                        continue
                    _hn = conf_seeds.iloc[_hi_idx].get(_name_col, f"#{_high} seed")
                    _ln = conf_seeds.iloc[_lo_idx].get(_name_col, f"#{_low} seed")

                    # Use fuzzy lookup so standings nicknames match team stats full names
                    _hn_row = _resolve_team_row(_hn)
                    _ln_row = _resolve_team_row(_ln)

                    # Get full names for display (fallback to nickname)
                    _hn_full = _hn_row.iloc[0].get(_team_name_col, _hn) if not _hn_row.empty else _hn
                    _ln_full = _ln_row.iloc[0].get(_team_name_col, _ln) if not _ln_row.empty else _ln

                    _has_prob = False
                    _hp_s = _lp_s = _hp_g = None
                    if not _hn_row.empty and not _ln_row.empty:
                        _hn_net = _hn_row.iloc[0].get("NET_RATING")
                        _ln_net = _ln_row.iloc[0].get("NET_RATING")
                        if _hn_net is not None and _ln_net is not None:
                            _md = float(_hn_net) - float(_ln_net)
                            _hp_g = 1 / (1 + math.exp(-(_md / 7)))
                            _hp_s = _series_prob(_hp_g)
                            _lp_s = 1 - _hp_s
                            _has_prob = True

                    _label = f"#{_high} {_hn} vs #{_low} {_ln}"
                    if _has_prob:
                        _fav = _hn_full if _hp_s >= 0.5 else _ln_full
                        _fav_prob = _hp_s if _hp_s >= 0.5 else _lp_s
                        _pred = _predicted_series(_fav.split()[-1], _fav_prob)
                        _label += f"  —  {_hn.split()[-1]} {_hp_s:.0%} / {_ln.split()[-1]} {_lp_s:.0%}  · Pred: {_pred}"

                    with st.expander(_label):
                        if _has_prob:
                            _bar_col1, _bar_col2 = st.columns(2)
                            with _bar_col1:
                                st.metric(f"{_hn_full} series win prob", f"{_hp_s:.1%}")
                                st.progress(_hp_s)
                            with _bar_col2:
                                st.metric(f"{_ln_full} series win prob", f"{_lp_s:.1%}")
                                st.progress(_lp_s)
                            _fav_full = _hn_full if _hp_s >= 0.5 else _ln_full
                            _fav_p = _hp_s if _hp_s >= 0.5 else _lp_s
                            _prediction = _predicted_series(_fav_full, _fav_p)
                            st.markdown(
                                f'<div style="background:#1A2035; border-left:4px solid #F0A500; '
                                f'padding:8px 14px; border-radius:4px; margin:8px 0;">'
                                f'<b style="color:#F0A500;">Prediction:</b> '
                                f'<span style="color:#FAFAFA; font-size:1.05em;">{_prediction}</span></div>',
                                unsafe_allow_html=True,
                            )
                            st.caption(f"Per-game: {_hn_full.split()[-1]} {_hp_g:.1%} / {_ln_full.split()[-1]} {1-_hp_g:.1%}")
                        else:
                            st.info("Net Rating data unavailable — team name may not match between standings and team stats.")

                        # Keys to the series button
                        _btn_key = f"keys_{_hn}_{_ln}".replace(" ", "_")
                        _report_key = f"keys_report_{_hn}_{_ln}".replace(" ", "_")
                        if st.button("Generate Keys to the Series", key=_btn_key,
                                     disabled=not st.session_state.api_key):
                            _hn_stats = _hn_row.iloc[0].to_dict() if not _hn_row.empty else {}
                            _ln_stats = _ln_row.iloc[0].to_dict() if not _ln_row.empty else {}
                            # Fetch rosters for both teams in this matchup
                            _hn_roster = _ensure_roster(_hn_full)
                            _ln_roster = _ensure_roster(_ln_full)
                            _hn_roster_names = _roster_player_names(_hn_roster)
                            _ln_roster_names = _roster_player_names(_ln_roster)
                            with st.spinner("Generating matchup keys…"):
                                _keys_report = generate_playoff_matchup_keys(
                                    _hn_full, _ln_full, _hn_stats, _ln_stats,
                                    _high, _low,
                                    _hp_s if _has_prob else 0.5,
                                    graph, st.session_state.api_key,
                                    roster_t1=_hn_roster_names,
                                    roster_t2=_ln_roster_names,
                                )
                            st.session_state[_report_key] = _keys_report

                        if st.session_state.get(_report_key):
                            st.markdown("---")
                            st.markdown(
                                f'<div class="report-box">{st.session_state[_report_key]}</div>',
                                unsafe_allow_html=True,
                            )

            _fr_c1, _fr_c2 = st.columns(2)
            with _fr_c1:
                _show_matchups(_east, "Eastern Conference")
            with _fr_c2:
                _show_matchups(_west, "Western Conference")
        else:
            st.markdown(
                '<div class="info-box">Could not parse standings columns. '
                'Playoff projections unavailable.</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ===========================================================
    # Section 3: Schedule Strength
    # ===========================================================
    st.markdown('<div class="section-header">Schedule Strength</div>', unsafe_allow_html=True)

    _sos_cols = st.columns(2)
    for _ti, (_tname, _trow) in enumerate([(team1_sel, _t1), (team2_sel, _t2)]):
        with _sos_cols[_ti]:
            st.markdown(f'<div class="player-badge">{_tname}</div>', unsafe_allow_html=True)

            # Home / Away record from standings
            if _sdf is not None and not _sdf.empty:
                _name_col_s = next((c for c in ["TeamName", "Team", "TEAM_NAME"] if c in _sdf.columns), None)
                if _name_col_s:
                    _team_std = _sdf[_sdf[_name_col_s] == _tname]
                    if not _team_std.empty:
                        _ts = _team_std.iloc[0]
                        _home_w = _ts.get("HOME_W") or _ts.get("HomeWin") or _ts.get("HOME_WINS")
                        _home_l = _ts.get("HOME_L") or _ts.get("HomeLoss") or _ts.get("HOME_LOSSES")
                        _road_w = _ts.get("ROAD_W") or _ts.get("AwayWin") or _ts.get("ROAD_WINS")
                        _road_l = _ts.get("ROAD_L") or _ts.get("AwayLoss") or _ts.get("ROAD_LOSSES")
                        _wins = _ts.get("WINS") or _ts.get("Win") or _ts.get("W")
                        _losses = _ts.get("LOSSES") or _ts.get("Loss") or _ts.get("L")

                        _items = {}
                        if _wins is not None and _losses is not None:
                            try:
                                _wf, _lf = float(_wins), float(_losses)
                                _items["Record"] = f"{int(_wf)}-{int(_lf)}"
                                _items["Win %"] = f"{_wf / (_wf + _lf):.1%}" if (_wf + _lf) > 0 else "—"
                            except Exception:
                                pass
                        if _home_w is not None and _home_l is not None:
                            try:
                                _items["Home Record"] = f"{int(float(_home_w))}-{int(float(_home_l))}"
                            except Exception:
                                pass
                        if _road_w is not None and _road_l is not None:
                            try:
                                _items["Away Record"] = f"{int(float(_road_w))}-{int(float(_road_l))}"
                            except Exception:
                                pass

                        if _items:
                            st.markdown(_stat_card("Season Record", _items), unsafe_allow_html=True)

            # SOS approximation: average opponent win% = average of all OTHER teams' win%
            _wins_col_t = next((c for c in ["WINS", "W"] if c in _tdf.columns), None)
            _losses_col_t = next((c for c in ["LOSSES", "L"] if c in _tdf.columns), None)
            if _wins_col_t and _losses_col_t:
                _other = _tdf[_tdf[_team_name_col] != _tname]
                _opp_wl = []
                for _, _or in _other.iterrows():
                    try:
                        _ow = float(_or[_wins_col_t])
                        _ol = float(_or[_losses_col_t])
                        if _ow + _ol > 0:
                            _opp_wl.append(_ow / (_ow + _ol))
                    except Exception:
                        pass
                if _opp_wl:
                    _avg_opp_wpct = sum(_opp_wl) / len(_opp_wl)
                    st.metric(
                        "Avg Opponent Win% (SOS estimate)",
                        f"{_avg_opp_wpct:.3f}",
                        help="Simple SOS: average win% of all other teams in the league (approximate)",
                    )
            else:
                # Try from standings
                if _sdf is not None and not _sdf.empty:
                    _wins_col_s = next((c for c in ["WINS", "Win", "W"] if c in _sdf.columns), None)
                    _losses_col_s = next((c for c in ["LOSSES", "Loss", "L"] if c in _sdf.columns), None)
                    _name_col_s2 = next((c for c in ["TeamName", "Team", "TEAM_NAME"] if c in _sdf.columns), None)
                    if _wins_col_s and _losses_col_s and _name_col_s2:
                        _other_s = _sdf[_sdf[_name_col_s2] != _tname]
                        _opp_wl_s = []
                        for _, _osr in _other_s.iterrows():
                            try:
                                _ow = float(_osr[_wins_col_s])
                                _ol = float(_osr[_losses_col_s])
                                if _ow + _ol > 0:
                                    _opp_wl_s.append(_ow / (_ow + _ol))
                            except Exception:
                                pass
                        if _opp_wl_s:
                            _avg_opp_wpct_s = sum(_opp_wl_s) / len(_opp_wl_s)
                            st.metric(
                                "Avg Opponent Win% (SOS estimate)",
                                f"{_avg_opp_wpct_s:.3f}",
                                help="Simple SOS: average win% of all other teams (approximate)",
                            )

            # Net Rating as supplementary context
            _nr = _trow.get("NET_RATING")
            if _nr is not None:
                st.metric("Net Rating", f"{float(_nr):+.1f}")


