"""
NBA Offensive–Defensive Player Matchup Network
SI 507 Final Project — Streamlit Application

Four interaction modes:
  1. Matchup Lookup        — head-to-head stats for any offensive vs. defensive player pair
  2. Player Profile        — full matchup neighborhood + bio + Basketball Reference stats
  3. Defensive Similarity  — find defenders with the most similar matchup profiles
  4. LLM Scouting Report  — Anthropic-powered narrative scouting reports
"""
import time
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict

from data_loader import (
    CACHE_DIR,
    enrich_graph,
    find_nba_player,
    get_player_bio,
    get_player_career_splits,
    get_player_shot_chart,
    get_player_shot_zones,
    get_team_head_coach,
    get_team_h2h_record,
    get_team_roster,
    load_matchup_data,
)
from bbref_loader import (
    get_bbref_team_stats,
    get_bbref_playoff_bracket,
)
from nba_api.stats.static import players as _nba_players_static
from nba_api.stats.static import teams as _nba_teams_static
try:
    from bbref_loader import (
        get_current_season_logs,
        get_playoff_logs,
        fmt_game_log_context,
    )
    _BBREF_AVAILABLE = True
except ImportError:
    _BBREF_AVAILABLE = False

from analyst_context import (
    detect_teams,
    detect_concepts,
    get_team_players,
    resolve_concept_players,
    fmt_player_compact,
)
from llm_reports import (
    ANALYST_SYSTEM_PROMPT,
    fmt_career_context,
    fmt_current_season_context,
    generate_matchup_report,
    generate_player_profile_report,
    generate_playoff_matchup_keys,
    generate_similarity_report,
    generate_team_matchup_report,
    stream_matchup_report,
    stream_player_profile_report,
    _call_anthropic,
)
from models import MatchupGraph
from visualizations import (
    plot_degree_distribution,
    plot_matchup_comparison,
    plot_neighborhood_bars,
    plot_network_neighborhood,
    plot_player_stats_bar,
    plot_ppp_heatmap,
    plot_shot_chart,
    plot_shot_chart_zones,
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
    call_cp_analysis_batch,
    call_cp_briefing,
    call_cp_qa,
    compute_drift,
    generate_example_questions,
    get_cross_team_matchups,
)

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
# Global CSS  (dark premium sports-analytics aesthetic)
# ---------------------------------------------------------------------------
_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800'
    '&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">'
)
_CSS = """
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0a0e17;
        color: #f1f5f9;
        font-family: 'DM Sans', 'Inter', Arial, sans-serif;
    }
    h1, h2, h3, h4 { font-family: 'DM Sans', Arial, sans-serif; }
    [data-testid="stSidebar"] {
        background-color: #0d1220;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div { color: #f1f5f9 !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] div { color: #111111 !important; }
    [data-testid="stMetric"] {
        background: #131a2b;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 14px 18px;
        transition: box-shadow 0.15s ease;
    }
    [data-testid="stMetric"]:hover { box-shadow: 0 0 0 1px rgba(59,130,246,0.3); }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; color: #f59e0b !important; font-family: 'JetBrains Mono', monospace !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #131a2b;
        padding: 6px 8px;
        border-radius: 12px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #475569;
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.88rem;
        font-family: 'DM Sans', Arial, sans-serif;
        border-bottom: 3px solid transparent;
        transition: color 0.15s ease, background 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #94a3b8 !important; }
    .stTabs [aria-selected="true"] {
        background: #1a2340 !important;
        color: #f1f5f9 !important;
        border-bottom: 3px solid #3b82f6 !important;
    }
    .dataframe-container { border-radius: 10px; overflow: hidden; }
    [data-testid="stDataFrame"] { border: 1px solid #1e293b; border-radius: 10px; }
    .stat-card {
        background: #131a2b;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 8px;
        transition: box-shadow 0.15s ease;
    }
    .stat-card:hover { box-shadow: 0 0 0 1px rgba(59,130,246,0.25); }
    .stat-card h4 {
        color: #f59e0b;
        margin: 0 0 10px 0;
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-family: 'DM Sans', Arial, sans-serif;
    }
    .stat-card p { margin: 3px 0; color: #94a3b8; font-size: 0.9rem; }
    .stat-card span.value { color: #f1f5f9; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
    .report-box {
        background: #131a2b;
        border: 1px solid #1e293b;
        border-left: 4px solid #f59e0b;
        border-radius: 12px;
        padding: 22px 26px;
        line-height: 1.75;
        font-size: 0.95rem;
        color: #e2e8f0;
        white-space: pre-wrap;
        font-family: 'DM Sans', Arial, sans-serif;
    }
    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f59e0b;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 8px;
        margin: 20px 0 14px 0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-family: 'DM Sans', Arial, sans-serif;
    }
    .player-badge {
        display: inline-block;
        background: #1e3a5f;
        color: #f1f5f9;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 5px 14px;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 8px;
        font-family: 'DM Sans', Arial, sans-serif;
    }
    .info-box {
        background: #0d1220;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 12px 16px;
        color: #94a3b8;
        font-size: 0.88rem;
    }
    .player-card {
        display: flex;
        align-items: center;
        gap: 16px;
        background: #131a2b;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .player-card-info { flex: 1; }
    .player-card-name { font-size: 1.15rem; font-weight: 700; color: #f1f5f9; }
    .player-card-meta { font-size: 0.82rem; color: #94a3b8; margin-top: 2px; }
    .matchup-vs-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 24px;
        background: #131a2b;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 16px;
    }
    .matchup-vs-divider { color: #475569; font-size: 1.2rem; font-weight: 600; font-family: 'DM Sans', Arial, sans-serif; }
    .matchup-player-block { text-align: center; }
    .matchup-player-block .name { font-size: 1.05rem; font-weight: 700; color: #f1f5f9; margin-top: 6px; }
    .matchup-player-block .meta { font-size: 0.78rem; color: #94a3b8; }
    hr { border-color: #1e293b !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .cp-entry {
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: box-shadow 0.15s ease;
    }
    .cp-entry:hover { box-shadow: 0 0 0 1px rgba(245,158,11,0.25); }
    .cp-entry-header { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
    .cp-entry .cp-player-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 6px;
        font-family: 'DM Sans', Arial, sans-serif;
    }
    .cp-entry .cp-narrative { color: #94a3b8; font-size: 0.87rem; margin: 4px 0; line-height: 1.55; }
    .cp-entry .cp-numbers { font-size: 0.87rem; margin: 4px 0; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
    .cp-entry .cp-coaching { color: #e2e8f0; font-size: 0.87rem; margin: 4px 0; font-style: italic; line-height: 1.55; }
    .cp-flag-callout {
        background: #130f00;
        border: 1px solid #1e293b;
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 12px 0;
        font-size: 0.88rem;
    }
    .cp-badge {
        display: inline-block;
        background: #f59e0b;
        color: #0a0e17;
        font-weight: 800;
        font-size: 0.7rem;
        border-radius: 4px;
        padding: 2px 7px;
        margin-right: 8px;
        letter-spacing: 0.06em;
        vertical-align: middle;
        font-family: 'DM Sans', Arial, sans-serif;
    }
    .cp-briefing {
        background: #0d1220;
        border: 1px solid #1e293b;
        border-left: 4px solid #3b82f6;
        border-radius: 12px;
        padding: 22px 26px;
        line-height: 1.8;
        font-size: 0.95rem;
        color: #e2e8f0;
        white-space: pre-wrap;
    }
    .cp-response-card {
        background: #131a2b;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px 22px;
        margin-top: 10px;
        transition: box-shadow 0.15s ease;
    }
    .cp-response-card:hover { box-shadow: 0 0 0 1px rgba(59,130,246,0.2); }
    .cp-response-header {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 10px;
        font-family: 'DM Sans', Arial, sans-serif;
    }
    .cp-leaderboard-row {
        background: #131a2b;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        font-size: 0.88rem;
        transition: box-shadow 0.15s ease;
    }
    .cp-leaderboard-row:hover { box-shadow: 0 0 0 1px rgba(245,158,11,0.25); }
    .stat-mono { font-family: 'JetBrains Mono', monospace; font-weight: 500; }
    @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
    .cp-loading { animation: pulse 1.4s ease-in-out infinite; color: #475569; font-style: italic; font-size: 0.85rem; }
"""
st.markdown(_FONTS, unsafe_allow_html=True)
st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


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
        "cp_ai_text": {},            # {player_id: {narrative, numbers_say, coaching_implication}}
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
        return "#ef4444"
    if ppp < avg - 0.15:
        return "#10b981"
    return "#f59e0b"


def _headshot_html(player_id: int, player_name: str, width: int = 80, height: int = 60) -> str:
    """
    Return an HTML <img> tag pointing to the NBA CDN headshot for player_id.
    On load failure (404 or timeout) the image is hidden and replaced with
    a styled initials placeholder in the primary accent colour.
    """
    url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
    initials = "".join(w[0].upper() for w in player_name.split()[:2] if w)
    font_size = max(10, width // 4)
    return (
        f'<span style="display:inline-block;position:relative;'
        f'width:{width}px;height:{height}px;flex-shrink:0;">'
        f'<img src="{url}" width="{width}" height="{height}" '
        f'style="border-radius:8px;border:1px solid #1e293b;object-fit:cover;'
        f'display:block;vertical-align:middle;box-shadow:0 2px 8px rgba(0,0,0,.4);" '
        f'onerror="this.style.display=\'none\';'
        f'this.nextElementSibling.style.display=\'flex\';">'
        f'<span style="display:none;width:{width}px;height:{height}px;'
        f'background:#131a2b;border:1px solid #1e293b;border-radius:8px;'
        f'align-items:center;justify-content:center;'
        f'font-size:{font_size}px;font-weight:700;color:#3b82f6;'
        f'font-family:\'DM Sans\',Arial,sans-serif;">{initials}</span>'
        f'</span>'
    )


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
        career_df, weighted_baseline = get_player_career_splits(player_id)
        result = compute_drift(player_id, career_df, weighted_baseline,
                               st.session_state.season, player_name=player_name)
    except Exception:
        result = None
    cache[player_id] = result
    return result


def _get_career_df_fast(player_id: int):
    """
    Return (career_df, weighted_baseline) without making any live NBA API calls.
    Check order: CounterPoint session cache → processed file cache → raw file cache.
    Returns (None, {}) if no cache exists so callers fall back gracefully.
    """
    # 1. Already fetched during CounterPoint analysis this session
    cached_df = st.session_state.get("cp_career_dfs", {}).get(player_id)
    if cached_df is not None and not cached_df.empty:
        return cached_df, {}

    # 2. Processed file cache (fast JSON read)
    proc_cache = CACHE_DIR / f"career_splits_processed_{player_id}.json"
    raw_cache  = CACHE_DIR / f"career_splits_{player_id}.json"
    if proc_cache.exists() or raw_cache.exists():
        try:
            return get_player_career_splits(player_id)
        except Exception:
            pass

    # 3. No cache at all — skip to avoid blocking for 60+ seconds on Streamlit Cloud
    return None, {}


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

    # Use AI-generated coaching implication if available, fall back to template
    _ai_entry = st.session_state.cp_ai_text.get(player_id, {})
    coaching_text = _ai_entry.get("coaching_implication") or drift["coaching_impl"]

    st.markdown(
        f'<div class="cp-flag-callout">'
        f'<span class="cp-badge">⚡ CP</span>'
        f'<span style="color:{color}; font-weight:700;">{label}</span>'
        f'&nbsp;&nbsp;'
        f'<span style="color:#e2e8f0; font-family:\'JetBrains Mono\',monospace;">'
        f'{slbl}: {career_str} {arrow} {curr_str}</span>'
        f'<br><br>'
        f'<span style="color:#94a3b8;">{coaching_text}</span>'
        f'<br>'
        f'<span style="color:#f59e0b; font-size:0.82rem; font-style:italic;">'
        f'Full analysis → CounterPoint tab</span>'
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
                if df.empty and season_type == "Playoffs":
                    st.warning(
                        "The NBA stats API does not provide player matchup data "
                        "(LeagueSeasonMatchups) for the Playoffs. "
                        "Try **Regular Season** for full matchup graph features. "
                        "Shot charts, player profiles, and team stats still work with Playoffs selected."
                    )
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

_playoff_no_matchups = (
    st.session_state.data_loaded
    and graph is not None
    and graph.graph.number_of_nodes() == 0
    and st.session_state.get("season_type") == "Playoffs"
)
_PLAYOFF_MATCHUP_WARNING = (
    '<div class="info-box" style="padding:20px;">'
    "<b style='color:#F0A500;'>Playoff Matchup Data Unavailable</b><br>"
    "The NBA Stats API does not publish player-vs-player matchup data for the Playoffs. "
    "Switch to <b>Regular Season</b> in the sidebar to use this feature."
    "</div>"
)

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

    if _playoff_no_matchups:
        st.markdown(_PLAYOFF_MATCHUP_WARNING, unsafe_allow_html=True)
    all_names = sorted(set(graph.all_player_names("offense")) | set(graph.all_player_names("defense"))) if not _playoff_no_matchups else []
    off_names = graph.all_player_names("offense") if not _playoff_no_matchups else []
    def_names = graph.all_player_names("defense") if not _playoff_no_matchups else []

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

            # Header with headshots
            _off_hs = _headshot_html(off_pid, off_player.name, 100, 75) if off_pid else ""
            _def_hs = _headshot_html(def_pid, def_player.name, 100, 75) if def_pid else ""
            _off_meta = " · ".join(filter(None, [off_player.position, off_player.team]))
            _def_meta = " · ".join(filter(None, [def_player.position, def_player.team]))
            st.markdown(
                f'<div class="matchup-vs-header">'
                f'<div class="matchup-player-block">'
                f'{_off_hs}'
                f'<div class="name">{off_player.name}</div>'
                f'<div class="meta">{_off_meta}</div>'
                f'</div>'
                f'<div class="matchup-vs-divider">vs</div>'
                f'<div class="matchup-player-block">'
                f'{_def_hs}'
                f'<div class="name">{def_player.name}</div>'
                f'<div class="meta">{_def_meta}</div>'
                f'</div>'
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
                _raw = edge.to_dict()
                _stats_df = pd.DataFrame(
                    {"Stat": list(_raw.keys()), "Value": [str(v) for v in _raw.values()]}
                )
                st.dataframe(_stats_df, hide_index=True, use_container_width=True)

            # ── Multi-season career matchup history ───────────────────────────
            st.markdown("---")
            st.markdown(
                '<div class="section-header">Career Matchup History</div>',
                unsafe_allow_html=True,
            )
            _seasons_to_check = ["2025-26", "2024-25", "2023-24", "2022-23"]
            _hist_rows = []
            for _hs in _seasons_to_check:
                _csv = CACHE_DIR / f"matchups_{_hs.replace('-','_')}_Regular_Season.csv"
                if not _csv.exists():
                    continue
                try:
                    _hdf = pd.read_csv(_csv)
                    _pair = _hdf[
                        (_hdf["OFF_PLAYER_ID"] == off_pid) &
                        (_hdf["DEF_PLAYER_ID"] == def_pid)
                    ]
                    if _pair.empty:
                        _hist_rows.append({"Season": _hs, "Poss": "—", "PPP": "—", "FG%": "—", "3PM/A": "—", "_poss_raw": 0})
                    else:
                        _hr = _pair.iloc[0]
                        _poss = float(_hr.get("PARTIAL_POSS", 0) or 0)
                        _pts  = float(_hr.get("PLAYER_PTS", 0)  or 0)
                        _ppp  = _pts / _poss if _poss > 0 else None
                        _fgp  = float(_hr.get("MATCHUP_FG_PCT", 0) or 0)
                        _fg3m = int(_hr.get("MATCHUP_FG3M", 0) or 0)
                        _fg3a = int(_hr.get("MATCHUP_FG3A", 0) or 0)
                        _hist_rows.append({
                            "Season": _hs,
                            "Poss":   f"{_poss:.0f}" if _poss > 0 else "—",
                            "PPP":    f"{_ppp:.3f}"  if _ppp  is not None else "—",
                            "FG%":    f"{_fgp:.1%}"  if _poss > 0 else "—",
                            "3PM/A":  f"{_fg3m}/{_fg3a}" if _poss > 0 else "—",
                            "_poss_raw": _poss,
                        })
                except Exception:
                    continue

            if _hist_rows:
                # Weighted aggregate row (weighted by possessions)
                _total_poss = sum(r["_poss_raw"] for r in _hist_rows)
                _seasons_with_data = [r for r in _hist_rows if r["_poss_raw"] > 0]
                _n_seasons = len(_seasons_with_data)
                _hist_rows_with_agg = _hist_rows  # default: no agg row when no data

                if _total_poss > 0:
                    # Re-read CSVs to compute weighted PPP and FG%
                    _w_ppp_num = 0.0; _w_fg_num = 0.0
                    for _hs2 in _seasons_to_check:
                        _csv2 = CACHE_DIR / f"matchups_{_hs2.replace('-','_')}_Regular_Season.csv"
                        if not _csv2.exists():
                            continue
                        try:
                            _hdf2 = pd.read_csv(_csv2)
                            _pair2 = _hdf2[
                                (_hdf2["OFF_PLAYER_ID"] == off_pid) &
                                (_hdf2["DEF_PLAYER_ID"] == def_pid)
                            ]
                            if not _pair2.empty:
                                _hr2  = _pair2.iloc[0]
                                _p2   = float(_hr2.get("PARTIAL_POSS", 0) or 0)
                                _pt2  = float(_hr2.get("PLAYER_PTS", 0)   or 0)
                                _fg2  = float(_hr2.get("MATCHUP_FG_PCT", 0) or 0)
                                _w_ppp_num += _pt2
                                _w_fg_num  += _fg2 * _p2
                        except Exception:
                            continue
                    _w_ppp = _w_ppp_num / _total_poss if _total_poss else None
                    _w_fg  = _w_fg_num  / _total_poss if _total_poss else None

                    _agg_label = (
                        f"**Weighted career matchup — {_total_poss:.0f} total possessions "
                        f"across {_n_seasons} season{'s' if _n_seasons != 1 else ''}**"
                        if _total_poss >= 10
                        else f"**Small sample — {_total_poss:.0f} possessions. Directional only.**"
                    )
                    st.markdown(_agg_label)

                    # Aggregate row at the top in bold
                    _agg_row = {
                        "Season": "Weighted avg",
                        "Poss":   f"{_total_poss:.0f}",
                        "PPP":    f"{_w_ppp:.3f}" if _w_ppp is not None else "—",
                        "FG%":    f"{_w_fg:.1%}"  if _w_fg  is not None else "—",
                        "3PM/A":  "—",
                        "_poss_raw": _total_poss,
                        "_is_agg":   True,
                    }
                    _hist_rows_with_agg = [_agg_row] + _hist_rows

                _display_rows = [{k: v for k, v in r.items()
                                  if k not in ("_poss_raw", "_is_agg")} for r in _hist_rows_with_agg]
                _hist_df = pd.DataFrame(_display_rows)
                _current_season = st.session_state.get("season", "2025-26")

                def _highlight_hist(row):
                    styles = []
                    for col in row.index:
                        if row.name == 0:  # aggregate row
                            styles.append("font-weight: bold; background-color: rgba(245,158,11,0.12);")
                        elif row["Season"] == _current_season:
                            styles.append("font-weight: bold; background-color: rgba(59,130,246,0.18);")
                        else:
                            styles.append("")
                    return styles

                styled_hist = _hist_df.style.apply(_highlight_hist, axis=1)
                st.dataframe(styled_hist, hide_index=True, use_container_width=True)
            else:
                st.markdown(
                    '<div class="info-box">No cached season data available to build career matchup history. '
                    'Load additional seasons to populate this table.</div>',
                    unsafe_allow_html=True,
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

    if _playoff_no_matchups:
        st.info(
            "Matchup graph data is unavailable for Playoffs. "
            "Shot charts and bio are shown below. "
            "Matchup neighborhood and advanced stats require Regular Season data."
        )

    _pp_fallback_names = sorted([p["full_name"] for p in _nba_players_static.get_active_players()]) if _playoff_no_matchups else []
    _pp_names = graph.all_player_names() if not _playoff_no_matchups else _pp_fallback_names

    col_a, col_b = st.columns([3, 1])
    with col_a:
        player_sel = st.selectbox("Player", _pp_names, index=0, key="pp_player")
    with col_b:
        role_sel = st.radio("Role", ["offense", "defense"], horizontal=True, key="pp_role")

    if player_sel:
        pid = graph.find_player_id(player_sel)
        if pid is None and _playoff_no_matchups:
            _static_match = find_nba_player(player_sel)
            pid = int(_static_match["id"]) if _static_match else None
        player = graph.players.get(pid) if pid else None

        if not player and _playoff_no_matchups and pid:
            _bio = get_player_bio(pid)
            if _bio:
                st.markdown("---")
                _pp_hs = _headshot_html(pid, player_sel, 200, 150)
                _pp_meta = " · ".join(filter(None, [
                    _bio.get("position"), _bio.get("team"), _bio.get("height"),
                    f"{_bio.get('weight')} lbs" if _bio.get("weight") else None
                ]))
                st.markdown(
                    f'<div style="margin-bottom:12px;">{_pp_hs}</div>'
                    f'<div class="player-badge" style="font-size:1.15rem;">{player_sel}</div>'
                    f'<div style="color:#94a3b8;font-size:0.85rem;margin:4px 0 10px;">{_pp_meta}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(_stat_card("Player Bio", {k: v for k, v in _bio.items() if v}), unsafe_allow_html=True)
                st.markdown("---")
                st.markdown("**Shot Chart (Playoffs)**")
                _sc_key = f"sc_{pid}_{st.session_state.get('season','2025-26')}_Playoffs"
                if _sc_key not in st.session_state:
                    with st.spinner("Loading shot chart…"):
                        _sc_df = get_player_shot_chart(pid, st.session_state.get("season", "2025-26"), "Playoffs")
                        st.session_state[_sc_key] = _sc_df
                _sc_df = st.session_state.get(_sc_key, pd.DataFrame())
                if not _sc_df.empty:
                    st.plotly_chart(plot_shot_chart(_sc_df, player_sel), use_container_width=True)
                else:
                    st.info("No shot chart data available for this player in the playoffs.")
        elif not player:
            st.warning("Player not found in graph.")
        else:
            st.markdown("---")

            # ---- Bio + stats row ----
            bio_col, stats_col, adv_col = st.columns([1.2, 1.4, 1.2])

            with bio_col:
                # Headshot + name card
                _pp_hs = _headshot_html(pid, player.name, 200, 150) if pid else ""
                _pp_meta = " · ".join(filter(None, [player.position, player.team,
                                                    player.height,
                                                    f"{player.weight} lbs" if player.weight else None]))
                st.markdown(
                    f'<div style="margin-bottom:12px;">{_pp_hs}</div>'
                    f'<div class="player-badge" style="font-size:1.15rem;">{player.name}</div>'
                    f'<div style="color:#94a3b8;font-size:0.85rem;margin:4px 0 10px;">{_pp_meta}</div>',
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

            # ---- Shot Zone Chart ----
            if pid:
                st.markdown("---")
                st.markdown(
                    f'<div class="section-header">{player.name} — Shot Zone Chart</div>',
                    unsafe_allow_html=True,
                )
                _sc_zone_key = f"zones_{pid}_{st.session_state.get('season','2025-26')}_{st.session_state.get('season_type','Regular Season')}"
                if _sc_zone_key not in st.session_state:
                    with st.spinner("Loading shot zones…"):
                        st.session_state[_sc_zone_key] = get_player_shot_zones(
                            int(pid),
                            season=st.session_state.get("season", "2025-26"),
                            season_type=st.session_state.get("season_type", "Regular Season"),
                        )
                _zone_summary = st.session_state[_sc_zone_key]
                if _zone_summary:
                    sc1, sc2 = st.columns([2, 1])
                    with sc1:
                        st.plotly_chart(
                            plot_shot_chart_zones(_zone_summary, player.name),
                            use_container_width=True,
                            key=f"zone_chart_{pid}",
                        )
                    with sc2:
                        st.markdown(
                            '<h4 style="color:#F0A500;font-size:0.85rem;text-transform:uppercase;'
                            'letter-spacing:0.08em;">Zone Breakdown</h4>',
                            unsafe_allow_html=True,
                        )
                        _zone_rows = []
                        for _zone, _zs in sorted(_zone_summary.items(), key=lambda x: -x[1]["freq"]):
                            _zone_rows.append({
                                "Zone":  _zone,
                                "FGA":   _zs["fga"],
                                "FGM":   _zs["fgm"],
                                "FG%":   f"{_zs['pct']:.1%}",
                                "Freq":  f"{_zs['freq']:.0%}",
                            })
                        st.dataframe(
                            pd.DataFrame(_zone_rows), hide_index=True, use_container_width=True
                        )
                        st.caption(
                            "Color key — green: above league avg | amber: near avg | red: below avg"
                        )
                else:
                    st.info("Shot zone data not available for this player/season.")


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

    if _playoff_no_matchups:
        st.markdown(_PLAYOFF_MATCHUP_WARNING, unsafe_allow_html=True)
    def_names_all = graph.all_player_names("defense") if not _playoff_no_matchups else []
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

    if _playoff_no_matchups:
        st.markdown(_PLAYOFF_MATCHUP_WARNING, unsafe_allow_html=True)
    if not st.session_state.api_key:
        st.markdown(
            '<div class="info-box">⚠️ Enter your <b>Anthropic API key</b> in the sidebar to enable scouting reports.</div>',
            unsafe_allow_html=True,
        )

    report_type = st.radio(
        "Report Type",
        ["Matchup Report", "Player Profile Report", "Defensive Similarity Report", "Team Matchup Report", "Ask the Analyst"],
        horizontal=True,
    )

    if report_type == "Matchup Report":
        r1, r2 = st.columns(2)
        with r1:
            off_r = st.selectbox("Offensive Player", graph.all_player_names("offense") if not _playoff_no_matchups else [], key="lr_off")
        with r2:
            def_r = st.selectbox("Defensive Player", graph.all_player_names("defense") if not _playoff_no_matchups else [], key="lr_def")

        if st.button("Generate Matchup Report", type="primary", disabled=not st.session_state.api_key):
            edge = graph.get_matchup(off_r, def_r)
            off_pid = graph.find_player_id(off_r)
            def_pid = graph.find_player_id(def_r)
            off_p = graph.players.get(off_pid)
            def_p = graph.players.get(def_pid)

            if edge and off_p and def_p:
                with st.spinner(f"Fetching stats for {off_r} and {def_r}…"):
                    _season_key = st.session_state.get("season", "2025-26")
                    _stype_key  = st.session_state.get("season_type", "Regular Season")
                    _off_zones = get_player_shot_zones(off_pid, _season_key, _stype_key) if off_pid else {}
                    _def_zones = get_player_shot_zones(def_pid, _season_key, _stype_key) if def_pid else {}
                    _off_career_df, _ = _get_career_df_fast(off_pid)
                    _def_career_df, _ = _get_career_df_fast(def_pid)

                    # Team records from team_stats_df
                    _tdf_rpt = st.session_state.get("team_stats_df")
                    _off_team_rec = _def_team_rec = None
                    if _tdf_rpt is not None and not _tdf_rpt.empty:
                        def _lookup_rec(team_name):
                            if not team_name:
                                return None
                            nick = team_name.split()[-1]
                            row = _tdf_rpt[_tdf_rpt["TEAM_NAME"].str.lower().str.contains(nick.lower(), na=False)]
                            return row.iloc[0].to_dict() if not row.empty else None
                        _off_team_rec = _lookup_rec(off_p.team)
                        _def_team_rec = _lookup_rec(def_p.team)

                    # Head-to-head record between the two teams
                    _h2h_rec = None
                    _tid_map_rpt = {t["nickname"]: t["id"] for t in _nba_teams_static.get_teams()}
                    _off_tid = _tid_map_rpt.get((off_p.team or "").split()[-1]) if off_p.team else None
                    _def_tid = _tid_map_rpt.get((def_p.team or "").split()[-1]) if def_p.team else None
                    if _off_tid and _def_tid and _off_tid != _def_tid:
                        try:
                            _h2h_raw = get_team_h2h_record(_off_tid, _def_tid, season=_season_key)
                            # Reorder so team1 = off_player's team
                            if _h2h_raw.get("games") and _h2h_raw["games"] and \
                               _h2h_raw["games"][0].get("team_id") == _def_tid:
                                _h2h_rec = {
                                    "team1_wins": _h2h_raw["team2_wins"],
                                    "team2_wins": _h2h_raw["team1_wins"],
                                    "games": _h2h_raw["games"],
                                }
                            else:
                                _h2h_rec = _h2h_raw
                        except Exception:
                            pass

                st.markdown("---")
                st.markdown(f"### Scouting Report: {off_r} vs {def_r}")
                _rpt_placeholder = st.empty()
                _rpt_text = ""
                for _chunk in stream_matchup_report(
                    edge, off_p, def_p,
                    graph.get_offensive_neighborhood(off_r, top_n=8),
                    graph.get_defensive_neighborhood(def_r, top_n=8),
                    st.session_state.api_key,
                    off_shot_zones=_off_zones,
                    def_shot_zones=_def_zones,
                    off_career_df=_off_career_df,
                    def_career_df=_def_career_df,
                    off_team_record=_off_team_rec,
                    def_team_record=_def_team_rec,
                    h2h_record=_h2h_rec,
                ):
                    _rpt_text += _chunk
                    _rpt_placeholder.markdown(
                        f'<div class="report-box">{_rpt_text}▌</div>',
                        unsafe_allow_html=True,
                    )
                _rpt_placeholder.markdown(
                    f'<div class="report-box">{_rpt_text}</div>',
                    unsafe_allow_html=True,
                )
                # CounterPoint flags — show for both players if drift detected
                if off_pid:
                    _render_cp_flag(off_pid, off_r)
                if def_pid:
                    _render_cp_flag(def_pid, def_r)
            else:
                st.warning("No direct matchup found for that pair with sufficient possessions.")

    elif report_type == "Player Profile Report":
        pp_r_player = st.selectbox("Player", graph.all_player_names() if not _playoff_no_matchups else [], key="lr_player")
        pp_r_role = st.radio("Role", ["offense", "defense"], horizontal=True, key="lr_role")

        if st.button("Generate Player Report", type="primary", disabled=not st.session_state.api_key):
            pid = graph.find_player_id(pp_r_player)
            player = graph.players.get(pid)
            if player:
                with st.spinner(f"Fetching stats for {pp_r_player}…"):
                    hood = (graph.get_offensive_neighborhood(pp_r_player, top_n=10)
                            if pp_r_role == "offense"
                            else graph.get_defensive_neighborhood(pp_r_player, top_n=10))
                    _pp_zones = get_player_shot_zones(
                        pid, st.session_state.get("season", "2025-26"),
                        st.session_state.get("season_type", "Regular Season"),
                    ) if pid else {}
                    _pp_career_df, _ = _get_career_df_fast(pid)

                st.markdown("---")
                st.markdown(f"### Scouting Report: {pp_r_player} ({pp_r_role.title()})")
                _pp_placeholder = st.empty()
                _pp_text = ""
                for _chunk in stream_player_profile_report(
                    player, pp_r_role, hood, st.session_state.api_key,
                    shot_zones=_pp_zones,
                    career_df=_pp_career_df,
                ):
                    _pp_text += _chunk
                    _pp_placeholder.markdown(
                        f'<div class="report-box">{_pp_text}▌</div>',
                        unsafe_allow_html=True,
                    )
                _pp_placeholder.markdown(
                    f'<div class="report-box">{_pp_text}</div>',
                    unsafe_allow_html=True,
                )
                if pid:
                    _render_cp_flag(pid, pp_r_player)
            else:
                st.warning("Player not found.")

    elif report_type == "Defensive Similarity Report":
        ds_r_def = st.selectbox("Defender", graph.all_player_names("defense") if not _playoff_no_matchups else [], key="lr_def2")

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

    elif report_type == "Ask the Analyst":
        st.markdown("Ask any basketball question and get a scout-quality answer.")

        analyst_question = st.text_area(
            "Your question",
            placeholder=(
                "e.g. How should a team defend a center who can shoot threes? "
                "What makes a good pick-and-roll defender? "
                "How do you attack a drop coverage big?"
            ),
            key="analyst_ask_input",
            height=100,
        )

        if st.button("Ask", type="primary", disabled=not st.session_state.api_key):
            if analyst_question.strip():
                _q_text = analyst_question.strip()
                _season = st.session_state.get("season", "2025-26")
                _season_end = int(_season.split("-")[0]) + 1

                # --- Layer 1: named player detection ---
                _all_nba = {p["full_name"].lower(): p for p in _nba_players_static.get_players()}
                _q_clean = _q_text.replace("'s", "").replace("'s", "")
                _words = _q_clean.split()
                _named_players = {}  # pid → name (explicitly mentioned)
                for _n in [3, 2]:
                    for _i in range(len(_words) - _n + 1):
                        _candidate = " ".join(_words[_i:_i+_n])
                        _clean = "".join(c for c in _candidate if c.isalpha() or c in " '-").strip()
                        if _clean.lower() in _all_nba and _all_nba[_clean.lower()]["id"] not in _named_players:
                            _named_players[_all_nba[_clean.lower()]["id"]] = _clean

                # --- Layer 2: team detection ---
                _detected_teams = detect_teams(_q_text) if graph else []

                # --- Layer 3: concept detection ---
                _detected_concepts = detect_concepts(_q_text)

                _career_parts = []
                _fetch_labels = []
                if _named_players:
                    _fetch_labels.append(", ".join(_named_players.values()))
                if _detected_teams:
                    _fetch_labels.append("teams: " + ", ".join(_detected_teams))
                if _detected_concepts:
                    _fetch_labels.append("concepts: " + ", ".join(_detected_concepts))

                _spinner_msg = ("Fetching data for " + " | ".join(_fetch_labels) + "...") if _fetch_labels else "Thinking..."

                with st.spinner(_spinner_msg):

                    # Named players -- full data + game logs + own team's head coach
                    _static_player_tid_map = {t["nickname"]: t["id"] for t in _nba_teams_static.get_teams()}
                    _injected_team_coaches: set = set()  # avoid duplication with team section
                    for _pid, _pname in list(_named_players.items())[:3]:
                        player_obj = graph.players.get(_pid) if graph else None
                        if player_obj:
                            _zones = get_player_shot_zones(_pid, _season, st.session_state.get("season_type", "Regular Season"))
                            _off_hood = graph.get_offensive_neighborhood(_pname, top_n=8)
                            _def_hood = graph.get_defensive_neighborhood(_pname, top_n=8)
                            _career_parts.append(fmt_current_season_context(player_obj, _zones, _off_hood, _def_hood))
                            # Inject head coach for the player's own team
                            _player_team = (player_obj.team or "").strip()
                            if _player_team and _player_team not in _injected_team_coaches:
                                _p_tid = _static_player_tid_map.get(_player_team)
                                if _p_tid:
                                    try:
                                        _p_coach = get_team_head_coach(
                                            _p_tid, season=_season
                                        )
                                        if _p_coach:
                                            _career_parts.append(f"{_player_team} head coach: {_p_coach}")
                                            _injected_team_coaches.add(_player_team)
                                    except Exception:
                                        pass
                        _cdf = None
                        for _attempt in range(3):
                            try:
                                _cdf, _ = get_player_career_splits(_pid)
                                if _cdf is not None and not _cdf.empty:
                                    break
                            except Exception:
                                pass
                            time.sleep(1)
                        if _cdf is not None and not _cdf.empty:
                            _career_parts.append(fmt_career_context(_cdf, _pname))
                        if _BBREF_AVAILABLE:
                            _reg_logs = get_current_season_logs(_pname, _season_end)
                            _playoff_logs = get_playoff_logs(_pname, _season_end)
                            _log_ctx = fmt_game_log_context(_pname, _reg_logs, _playoff_logs)
                            if _log_ctx:
                                _career_parts.append(_log_ctx)

                    # Team stats & roster — injected for every detected team
                    _tdf_analyst = st.session_state.get("team_stats_df")
                    # Static team ID lookup — always available, no data-load required
                    _static_tid_map = {t["nickname"]: t["id"] for t in _nba_teams_static.get_teams()}
                    for _team in _detected_teams:
                        _team_lines = [f"=== {_team.upper()} — TEAM STATS & ROSTER ==="]
                        # Resolve team_id: static map first, then session cache
                        _analyst_tid = (
                            _static_tid_map.get(_team)
                            or st.session_state.get("roster_team_ids", {}).get(_team)
                        )
                        # Team-level stats (available only after "Load Team Data")
                        if _tdf_analyst is not None and not _tdf_analyst.empty:
                            _t_row = _tdf_analyst[
                                _tdf_analyst["TEAM_NAME"].str.lower().str.contains(
                                    _team.split()[-1].lower(), na=False
                                )
                            ]
                            if not _t_row.empty:
                                _tr = _t_row.iloc[0]
                                def _tf(col):
                                    v = _tr.get(col)
                                    try:
                                        return f"{float(v):.1f}" if v is not None else "—"
                                    except (TypeError, ValueError):
                                        return "—"
                                _team_lines.append(
                                    f"Team stats: OffRtg {_tf('OFF_RATING')} | DefRtg {_tf('DEF_RATING')} | "
                                    f"NetRtg {_tf('NET_RATING')} | Pace {_tf('PACE')} | "
                                    f"W-L {int(_tr.get('W', 0) or 0)}-{int(_tr.get('L', 0) or 0)}"
                                )
                        # Head coach — fetched via NBA API (cached per team)
                        if _analyst_tid:
                            try:
                                _head_coach = get_team_head_coach(
                                    _analyst_tid, season=st.session_state.get("season", "2025-26")
                                )
                                if _head_coach:
                                    _team_lines.append(f"Head coach: {_head_coach}")
                            except Exception:
                                pass
                        # Roster players (available after matchup data is loaded)
                        _team_players = get_team_players(_team, graph) if graph else []
                        for _tp in _team_players:
                            if _tp.player_id not in _named_players:
                                _tp_cdf = None
                                try:
                                    _tp_cdf, _ = get_player_career_splits(_tp.player_id)
                                except Exception:
                                    pass
                                _team_lines.append(fmt_player_compact(_tp, _tp_cdf))
                        if len(_team_lines) > 1:
                            _career_parts.append("\n".join(_team_lines))

                    # Concept pools — enriched stats + improvement delta (compact)
                    _career_cache = {}
                    if _detected_concepts and graph:
                        # Pre-load career splits for concept resolution
                        for _p in graph.players.values():
                            if _p.player_id not in _career_cache:
                                try:
                                    _cdf2, _ = get_player_career_splits(_p.player_id)
                                    if _cdf2 is not None and not _cdf2.empty:
                                        _career_cache[_p.player_id] = _cdf2
                                except Exception:
                                    pass

                        for _concept in _detected_concepts:
                            _concept_players = resolve_concept_players(_concept, graph, _career_cache)
                            if _concept_players:
                                _concept_lines = [f"=== {_concept.upper().replace('_',' ')} CANDIDATES (ranked by current stats) ==="]
                                for _cp in _concept_players:
                                    if _cp.player_id not in _named_players:
                                        _concept_lines.append(fmt_player_compact(_cp, _career_cache.get(_cp.player_id)))
                                _career_parts.append("\n".join(_concept_lines))

                _career_context = ""
                if _career_parts:
                    _career_context = (
                        "\n\n=== VERIFIED PLAYER DATA (NBA Stats API + Basketball Reference) ===\n"
                        "Cite numbers from this section with full confidence. "
                        "If a stat you want is not here, state the claim qualitatively.\n\n"
                        + "\n\n".join(_career_parts)
                    )

                _data_preamble = ""
                if _career_context:
                    _data_preamble = (
                        f"{_career_context}\n\n"
                        f"The numbers above are verified facts from the NBA Stats API. "
                        f"For game planning questions, weight the CURRENT SEASON stats heavily — "
                        f"that is what the player is doing right now and what a coaching staff must prepare for. "
                        f"Career history is useful context for understanding if a trend is new or established, "
                        f"but it does not override this season's numbers. "
                        f"If a player's current season diverges from their career reputation, flag it explicitly "
                        f"and lead your answer with what they are doing now. "
                        f"Do not say a stat is unavailable if it appears in the table above.\n\n"
                    )

                prompt = (
                    f"{_data_preamble}"
                    f"Question: {_q_text}\n\n"
                    f"Answer this at the depth a coaching staff would expect from a senior scout. "
                    f"Be direct, use specific scheme language, and cite real examples where they sharpen the argument."
                )
                _summary_parts = []
                if _named_players:
                    _summary_parts.append("Players: " + ", ".join(_named_players.values()))
                if _detected_teams:
                    _summary_parts.append("Teams: " + ", ".join(_detected_teams))
                if _detected_concepts:
                    _summary_parts.append("Concepts: " + ", ".join(_detected_concepts))
                if _summary_parts:
                    st.caption("Data fetched for — " + " | ".join(_summary_parts))
                else:
                    st.caption("No players, teams, or concepts detected — answering from general knowledge")

                if _career_context:
                    with st.expander("Show injected career data (debug)", expanded=False):
                        st.code(_career_context, language=None)

                with st.spinner("The analyst is thinking…"):
                    _analyst_report = _call_anthropic(
                        prompt,
                        st.session_state.api_key,
                        system_override=ANALYST_SYSTEM_PROMPT,
                    )
                st.markdown("---")
                st.markdown("### The Analyst")
                st.markdown(f'<div class="report-box">{_analyst_report}</div>', unsafe_allow_html=True)
            else:
                st.warning("Enter a question first.")

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

    if _playoff_no_matchups:
        st.markdown(_PLAYOFF_MATCHUP_WARNING, unsafe_allow_html=True)
    else:
        summ = graph.get_summary()
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total Players", summ["total_nodes"])
        m2.metric("Offensive", summ["offensive_players"])
        m3.metric("Defensive", summ["defensive_players"])
        m4.metric("Matchup Edges", summ["total_edges"])
        m5.metric("Graph Density", f"{summ['density']:.4f}")
        m6.metric("Avg PPP", f"{summ['avg_ppp']:.3f}")

        st.markdown("---")

        off_degs, def_degs = graph.degree_sequences()
        st.plotly_chart(plot_degree_distribution(off_degs, def_degs), use_container_width=True)

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

        # Career DataFrame store so the panel can reuse what was fetched for drift scoring
        if "cp_career_dfs" not in st.session_state:
            st.session_state.cp_career_dfs = {}

        if _cp_analyse_btn or st.session_state.cp_matchup_drift:
            if _cp_analyse_btn:
                # ── Step 1: Compute narrative drift for all offensive players ──
                _cp_pids_to_run = list({m["off_pid"] for m in _cp_matchups})
                _cp_prog = st.progress(0, text="Computing narrative drift…")
                for _ci, _cpid in enumerate(_cp_pids_to_run):
                    _cp_prog.progress(
                        (_ci + 1) / max(len(_cp_pids_to_run), 1),
                        text=f"Analysing {graph.players[_cpid].name if _cpid in graph.players else _cpid}…",
                    )
                    # Retry on re-click if previously None (API may have been down)
                    _already_ok = (
                        _cpid in st.session_state.cp_matchup_drift
                        and st.session_state.cp_matchup_drift[_cpid] is not None
                    )
                    if not _already_ok:
                        try:
                            _cp_career_df, _cp_wb = get_player_career_splits(_cpid)
                            _cp_pname = graph.players[_cpid].name if _cpid in graph.players else ""
                            _cp_result = compute_drift(_cpid, _cp_career_df, _cp_wb,
                                                       st.session_state.season, player_name=_cp_pname)
                            st.session_state.cp_career_dfs[_cpid] = _cp_career_df
                        except Exception:
                            _cp_result = None
                        st.session_state.cp_matchup_drift[_cpid] = _cp_result
                _cp_prog.empty()

                # ── Step 2: Batch AI analysis for all flagged players ──────────
                if st.session_state.api_key:
                    _batch_flagged = []
                    for _m in _cp_matchups:
                        _bpid   = _m["off_pid"]
                        _bdrift = st.session_state.cp_matchup_drift.get(_bpid)
                        if _bdrift and _bdrift.get("flagged"):
                            _bpl = graph.players.get(_bpid)
                            _batch_entry = {
                                "name":     _m["off_player"],
                                "off_team": _m["off_team"],
                                "drift":    _bdrift,
                            }
                            if _bpl:
                                if _bpl.position:
                                    _batch_entry["position"] = _bpl.position
                                if _bpl.height:
                                    _batch_entry["height"] = _bpl.height
                                if _bpl.ppg is not None:
                                    _batch_entry["ppg"] = f"{_bpl.ppg:.1f}"
                                if _bpl.usg_pct is not None:
                                    _batch_entry["usage_rate"] = f"{_bpl.usg_pct:.1%}"
                            _batch_flagged.append(_batch_entry)

                    if _batch_flagged:
                        with st.spinner("Generating AI analysis for flagged players…"):
                            _ai_results = call_cp_analysis_batch(
                                _batch_flagged, _cp_t1, _cp_t2,
                                st.session_state.api_key,
                            )
                        for _m in _cp_matchups:
                            _bpid  = _m["off_pid"]
                            _pname = _m["off_player"]
                            if _pname in _ai_results:
                                st.session_state.cp_ai_text[_bpid] = _ai_results[_pname]

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
                # Track rendered pairs to prevent duplicate entries (dedup both directions)
                _rendered_pairs: set = set()

                for _mi, _m in enumerate(_cp_matchups):
                    _off_pid  = _m["off_pid"]
                    _def_pid  = _m["def_pid"]
                    _off_name = _m["off_player"]
                    _def_name = _m["def_player"]
                    _drift    = st.session_state.cp_matchup_drift.get(_off_pid)

                    # Skip if the reverse pair was already rendered
                    _pair_key = frozenset({_off_pid, _def_pid})
                    if _pair_key in _rendered_pairs:
                        continue
                    _rendered_pairs.add(_pair_key)

                    # Headshots
                    _off_hs = _headshot_html(_off_pid, _off_name, 48, 36)
                    _def_hs = _headshot_html(_def_pid, _def_name, 48, 36)

                    # ── Helper: build mini stat table rows (4 max: current + 3 prior) ──
                    def _mini_table_rows(stats_for_stat, traj_dict, trend_dirs, stat_key, slabel):
                        """Return list of (season, value_str, arrow) for the mini table."""
                        seasons = traj_dict.get("seasons", [])
                        values  = traj_dict.get("values",  [])
                        if not seasons:
                            return []
                        arrow_map = {"up": "↑", "down": "↓", "flat": "→"}
                        trend = trend_dirs.get(stat_key, "flat")
                        arrow = arrow_map.get(trend, "→")
                        fmt = ".1%" if ("pct" in stat_key or stat_key == "ft_rate") else ".1f"
                        rows_out = []
                        for s, v in zip(seasons[-4:], values[-4:]):  # last 4, ascending
                            rows_out.append((s, f"{v:{fmt}}", arrow))
                        rows_out.reverse()  # most recent first for display
                        return rows_out

                    def _render_mini_table(table_rows, stat_label, key_suffix):
                        if not table_rows:
                            return
                        rows_html = "".join(
                            f'<tr>'
                            f'<td style="padding:2px 8px;color:#94a3b8;font-size:0.78rem;">{s}</td>'
                            f'<td style="padding:2px 8px;font-family:\'JetBrains Mono\',monospace;'
                            f'font-size:0.78rem;color:#e2e8f0;">{v}</td>'
                            f'<td style="padding:2px 8px;font-size:0.82rem;color:#f59e0b;">{a}</td>'
                            f'</tr>'
                            for s, v, a in table_rows
                        )
                        st.markdown(
                            f'<table style="border-collapse:collapse;margin:4px 0 8px 0;">'
                            f'<thead><tr>'
                            f'<th style="padding:2px 8px;color:#475569;font-size:0.72rem;'
                            f'text-align:left;">Season</th>'
                            f'<th style="padding:2px 8px;color:#475569;font-size:0.72rem;'
                            f'text-align:left;">{stat_label}</th>'
                            f'<th style="padding:2px 8px;color:#475569;font-size:0.72rem;'
                            f'text-align:left;">Trend</th>'
                            f'</tr></thead><tbody>{rows_html}</tbody></table>',
                            unsafe_allow_html=True,
                        )

                    if _drift and _drift.get("flagged"):
                        _flag   = _drift["flag"]
                        _color  = FLAG_COLOR.get(_flag, "#94a3b8")
                        _flabel = FLAG_LABEL.get(_flag, "")
                        _stat   = _drift["max_drift_stat"]
                        _slbl   = CP_STAT_LABELS.get(_stat, _stat)
                        _trend_dirs = _drift.get("trend_directions", {})

                        # Use AI-generated text if available, else fall back to template
                        _ai_entry  = st.session_state.cp_ai_text.get(_off_pid, {})
                        _narrative = _ai_entry.get("narrative") or _drift["narrative"]
                        _numbers   = _ai_entry.get("numbers_say") or _drift["numbers_say"]
                        _coaching  = _ai_entry.get("coaching_implication") or _drift["coaching_impl"]

                        _cp_flagged.append({
                            "name":     _off_name,
                            "off_team": _m["off_team"],
                            "drift":    _drift,
                        })

                        st.markdown(
                            f'<div class="cp-entry" style="background:#131a2b; '
                            f'border-left: 4px solid {_color};">'
                            f'<div class="cp-entry-header">'
                            f'{_off_hs}'
                            f'<div style="flex:1;">'
                            f'<div class="cp-player-name">{_off_name} '
                            f'<span style="color:#94a3b8; font-size:0.8rem; font-weight:400;">'
                            f'({_m["off_team"]}) vs {_def_name} ({_m["def_team"]})</span></div>'
                            f'<div style="display:inline-block; background:{_color}22; '
                            f'color:{_color}; border:1px solid {_color}55; border-radius:5px; '
                            f'padding:2px 9px; font-size:0.73rem; font-weight:700;">{_flabel}</div>'
                            f'</div>'
                            f'</div>'
                            f'<div class="cp-narrative"><b>The narrative:</b> {_narrative}</div>'
                            f'<div class="cp-numbers" style="color:{_color};">'
                            f'<b>The numbers say:</b> {_numbers}</div>'
                            f'<div class="cp-coaching"><b>Coaching implication:</b> {_coaching}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        # Sparkline for the flagged stat — unique key prevents duplicate chart error
                        _traj = _drift["trajectories"].get(_stat, {})
                        if _traj.get("seasons") and len(_traj["seasons"]) >= 2:
                            _spark_key = f"sparkline_{_off_pid}_{_stat}_{_mi}_panel"
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

                        # Mini stat table (flagged stat, 4 rows max)
                        _mt_rows = _mini_table_rows(
                            _drift.get("current_vals", {}), _traj, _trend_dirs, _stat, _slbl
                        )
                        _render_mini_table(_mt_rows, _slbl, f"{_off_pid}_{_stat}_{_mi}")

                    elif _drift and not _drift.get("flagged"):
                        # Stable player — show stable_summary + mini table for 2 most stable stats
                        _stable_txt = _drift.get("stable_summary", "")
                        _trend_dirs = _drift.get("trend_directions", {})
                        _ds_scores  = _drift.get("drift_scores", {})

                        # Most stable = lowest |z-score|
                        _stable_priority = [s for s in CP_STAT_LABELS if s in _ds_scores]
                        _stable_sorted   = sorted(_stable_priority, key=lambda s: abs(_ds_scores.get(s, 0)))
                        _stable_show     = _stable_sorted[:2]

                        st.markdown(
                            f'<div class="cp-entry" style="background:#131a2b; '
                            f'border-left: 4px solid #1e293b;">'
                            f'<div class="cp-entry-header">'
                            f'{_off_hs}'
                            f'<div class="cp-player-name">{_off_name} '
                            f'<span style="color:#94a3b8; font-size:0.8rem; font-weight:400;">'
                            f'({_m["off_team"]}) vs {_def_name} ({_m["def_team"]})</span></div>'
                            f'</div>'
                            f'<div style="color:#475569; font-size:0.85rem;">'
                            f'{_stable_txt if _stable_txt else "No narrative drift detected — conventional scouting is holding up."}'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        # Mini table for each of the 2 most stable stats
                        for _ss in _stable_show:
                            _ss_traj = _drift.get("trajectories", {}).get(_ss, {})
                            _ss_lbl  = CP_STAT_LABELS.get(_ss, _ss)
                            _smt_rows = _mini_table_rows(
                                _drift.get("current_vals", {}), _ss_traj, _trend_dirs, _ss, _ss_lbl
                            )
                            _render_mini_table(_smt_rows, _ss_lbl, f"{_off_pid}_{_ss}_{_mi}_stable")

                    else:
                        # _drift is None: either not yet computed, or computed with insufficient data
                        _was_tried = _off_pid in st.session_state.cp_matchup_drift
                        _no_data_msg = (
                            "Insufficient career history to compute narrative drift — fewer than 2 qualifying seasons on record."
                            if _was_tried else
                            "Click ⚡ Run CounterPoint Analysis to compute narrative drift for this matchup."
                        )
                        st.markdown(
                            f'<div class="cp-entry" style="background:#131a2b; '
                            f'border-left: 4px solid #1e293b;">'
                            f'<div class="cp-entry-header">'
                            f'{_off_hs}'
                            f'<div class="cp-player-name">{_off_name} '
                            f'<span style="color:#94a3b8; font-size:0.8rem; font-weight:400;">'
                            f'({_m["off_team"]}) vs {_def_name} ({_m["def_team"]})</span></div>'
                            f'</div>'
                            f'<div style="color:#475569; font-size:0.85rem;">'
                            f'{_no_data_msg}</div>'
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
                            _lb_career_df, _lb_wb = get_player_career_splits(_lp.player_id)
                            _lb_result = compute_drift(
                                _lp.player_id, _lb_career_df, _lb_wb,
                                st.session_state.season, player_name=_lp.name,
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
                        _col   = FLAG_COLOR.get(d["flag"], "#94a3b8")
                        _sl    = CP_STAT_LABELS.get(d["max_drift_stat"], d["max_drift_stat"])
                        _ca    = d["career_avgs"].get(d["max_drift_stat"])
                        _cv    = d["current_vals"].get(d["max_drift_stat"])
                        _fmt   = ".1%" if "pct" in d["max_drift_stat"] or d["max_drift_stat"] == "ft_rate" else ".1f"
                        _ca_s  = f"{_ca:{_fmt}}" if _ca is not None else "—"
                        _cv_s  = f"{_cv:{_fmt}}" if _cv is not None else "—"
                        _z     = d["max_drift_score"]
                        _lb_hs = _headshot_html(pid, _pl.name, 48, 36)
                        st.markdown(
                            f'<div class="cp-leaderboard-row" style="display:flex;align-items:center;gap:12px;">'
                            f'{_lb_hs}'
                            f'<div style="flex:1;">'
                            f'<span style="color:#475569; font-size:0.78rem;">#{rank}</span>&nbsp;'
                            f'<b style="color:#f1f5f9;">{_pl.name}</b>&nbsp;'
                            f'<span style="color:#94a3b8; font-size:0.8rem;">{_pl.team or ""}</span>'
                            f'&nbsp;&nbsp;'
                            f'<span style="background:{_col}22; color:{_col}; border:1px solid {_col}55; '
                            f'border-radius:5px; padding:2px 7px; font-size:0.73rem;">{_fl}</span>'
                            f'<br>'
                            f'<span style="color:#475569; font-size:0.8rem;">Driving stat: '
                            f'<b style="color:#94a3b8; font-family:\'JetBrains Mono\',monospace;">{_sl}</b>'
                            f' — career {_ca_s} → this season {_cv_s} '
                            f'({_z:+.1f}\u03c3)</span>'
                            f'</div>'
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
        _season_end = int(_season.split("-")[0]) + 1

        tdf = get_bbref_team_stats(_season_end)
        bracket = get_bbref_playoff_bracket(_season_end)

        # BBRef team stats doubles as standings (has Conference, PlayoffRank, W, L)
        # Add FULL_NAME alias so existing column-detection code still works
        if not tdf.empty:
            tdf["FULL_NAME"] = tdf["TEAM_NAME"]
            tdf["TeamName"] = tdf["TEAM_NAME"]

        st.session_state.team_stats_df = tdf if not tdf.empty else None
        st.session_state.standings_df = tdf if not tdf.empty else None
        st.session_state.playoff_bracket_list = bracket
        # Keep playoff_series_df as None — no longer used
        st.session_state.playoff_series_df = None
        st.session_state.team_data_loaded = True
        st.session_state.team_data_updated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if force:
            st.session_state.roster_cache = {}

        return []

    if load_team_btn:
        with st.spinner("Fetching live team stats and playoff bracket from Basketball Reference..."):
            _load_all_team_data(force=False)
        if st.session_state.team_stats_df is not None:
            st.success(f"Loaded stats for {len(st.session_state.team_stats_df)} teams.")

    if refresh_team_btn:
        with st.spinner("Re-fetching from Basketball Reference..."):
            _load_all_team_data(force=True)
        if st.session_state.team_stats_df is not None:
            st.success("Data refreshed from Basketball Reference.")

    if not st.session_state.team_data_loaded or st.session_state.team_stats_df is None:
        st.markdown(
            '<div class="info-box">Click <b>Load Team Data</b> above to fetch NBA team stats '
            'and standings for the selected season.</div>',
            unsafe_allow_html=True,
        )
    else:

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

        else:
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
            # Section 2: Playoff Bracket (live from Basketball Reference)
            # ===========================================================
            st.markdown('<div class="section-header">Playoff Bracket</div>', unsafe_allow_html=True)

            _bracket = st.session_state.get("playoff_bracket_list", [])

            if not _bracket:
                st.markdown(
                    '<div class="info-box">Playoff bracket unavailable. Click Load Team Data to fetch from Basketball Reference.</div>',
                    unsafe_allow_html=True,
                )
            else:
                import math

                def _series_prob(p: float) -> float:
                    """Best-of-7 series win probability given single-game win prob p."""
                    q = 1 - p
                    return p**4 * (1 + 4*q + 10*q**2 + 20*q**3)

                def _net_rating(team_name):
                    if _tdf is None:
                        return None
                    row = _tdf[_tdf["TEAM_NAME"].str.lower() == team_name.lower()]
                    if row.empty:
                        # try partial match
                        row = _tdf[_tdf["TEAM_NAME"].str.lower().str.contains(team_name.split()[-1].lower(), na=False)]
                    return float(row["NET_RATING"].iloc[0]) if not row.empty else None

                # Group by round
                _rounds_order = ["Finals", "Conference Finals", "Conference Semifinals", "First Round"]
                _by_round = {}
                for _s in _bracket:
                    _by_round.setdefault(_s["round_name"], []).append(_s)

                _status_icons = {"in_progress": "🔴", "upcoming": "🔜", "completed": "✅"}

                for _rname in _rounds_order:
                    _series_in_round = _by_round.get(_rname, [])
                    if not _series_in_round:
                        continue

                    # Detect if any series in this round is active
                    _has_active = any(s["status"] in ("in_progress", "upcoming") for s in _series_in_round)
                    _round_label = f"**{_rname}**" + (" — In Progress" if _has_active else "")
                    st.markdown(f"#### {_round_label}")

                    _cols = st.columns(min(len(_series_in_round), 2))
                    for _ci, _s in enumerate(_series_in_round):
                        with _cols[_ci % 2]:
                            _icon = _status_icons.get(_s["status"], "")
                            _score_str = f"{_s['wins1']}-{_s['wins2']}"

                            if _s["status"] == "completed":
                                _header = f"{_icon} **{_s['leader']}** def. {_s['team2'] if _s['leader'] == _s['team1'] else _s['team1']} ({_score_str})"
                            elif _s["status"] == "in_progress":
                                _leader_txt = f"{_s['leader']} leads" if _s["leader"] else "Tied"
                                _header = f"{_icon} **{_s['team1']}** vs **{_s['team2']}** — {_leader_txt} {_score_str}"
                            else:
                                _header = f"{_icon} **{_s['team1']}** vs **{_s['team2']}** — Upcoming"

                            st.markdown(_header)

                            # Series win probability for active series
                            if _s["status"] in ("in_progress", "upcoming") and _tdf is not None:
                                _nr1 = _net_rating(_s["team1"])
                                _nr2 = _net_rating(_s["team2"])
                                if _nr1 is not None and _nr2 is not None:
                                    _diff = _nr1 - _nr2
                                    _pg1 = 1 / (1 + math.exp(-(_diff / 7)))
                                    _ps1 = _series_prob(_pg1)
                                    _ps2 = 1 - _ps1
                                    st.caption(
                                        f"Series probability: {_s['team1'].split()[-1]} {_ps1:.0%} / "
                                        f"{_s['team2'].split()[-1]} {_ps2:.0%} "
                                        f"(Net Rtg: {_nr1:+.1f} vs {_nr2:+.1f})"
                                    )

                            # Game log expander
                            _played = [g for g in _s["games"] if g["played"]]
                            _sched = [g for g in _s["games"] if not g["played"]]
                            if _played or _sched:
                                with st.expander("Games", expanded=(_s["status"] == "in_progress")):
                                    for _g in _played:
                                        _winner = _g["home"] if _g["home_score"] > _g["away_score"] else _g["away"]
                                        st.caption(
                                            f"{_g['date']}: {_g['away']} {_g['away_score']} @ "
                                            f"{_g['home']} {_g['home_score']} "
                                            f"({'**W**' if _winner == _s['team1'] else 'L'})"
                                        )
                                    for _g in _sched[:3]:
                                        st.caption(f"{_g['date']}: {_g['away']} @ {_g['home']} (scheduled)")

                            # Keys to the series button for active matchups
                            if _s["status"] in ("in_progress", "upcoming"):
                                _key_btn_key = f"keys_{_s['team1']}_{_s['team2']}"
                                if st.session_state.api_key and st.button("Keys to the Series", key=_key_btn_key):
                                    _t1r = _tdf[_tdf["TEAM_NAME"].str.lower() == _s["team1"].lower()]
                                    _t2r = _tdf[_tdf["TEAM_NAME"].str.lower() == _s["team2"].lower()]
                                    _t1_stats = _t1r.iloc[0].to_dict() if not _t1r.empty else {}
                                    _t2_stats = _t2r.iloc[0].to_dict() if not _t2r.empty else {}
                                    with st.spinner("Generating keys to the series..."):
                                        _keys_report = generate_playoff_matchup_keys(
                                            _s["team1"], _s["team2"],
                                            _t1_stats, _t2_stats,
                                            graph, st.session_state.api_key,
                                        )
                                    if _keys_report:
                                        st.markdown(f'<div class="report-box">{_keys_report}</div>', unsafe_allow_html=True)
                            st.markdown("---")

            st.markdown("---")

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


