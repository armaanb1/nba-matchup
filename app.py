"""
The Matchup Lab — NBA offensive/defensive archetype analysis.
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
    run_archetype_classification,
    classify_archetypes_from_existing_stats,
)
from bbref_loader import (
    get_bbref_team_stats,
    get_bbref_playoff_bracket,
    fmt_playoff_context,
    get_playoff_series_boxscores,
    get_cached_series_boxscores_for_teams,
    prefetch_playoff_boxscores,
    get_team_season_results,
    fmt_team_season_results,
)
from nba_api.stats.static import players as _nba_players_static
from nba_api.stats.static import teams as _nba_teams_static
try:
    from bbref_loader import (
        get_current_season_logs,
        get_playoff_logs,
        fmt_game_log_context,
        prefetch_player_logs,
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
    SYSTEM_PROMPT,
    fmt_career_context,
    fmt_current_season_context,
    generate_matchup_report,
    generate_player_profile_report,
    generate_similarity_report,
    stream_matchup_report,
    stream_player_profile_report,
    _call_anthropic,
)
from models import MatchupGraph, OffensiveArchetype, DefensiveRole
from visualizations import (
    plot_matchup_comparison,
    plot_neighborhood_bars,
    plot_network_neighborhood,
    plot_player_stats_bar,
    plot_shot_chart,
    plot_shot_chart_zones,
    plot_similarity_comparison,
    plot_similarity_scores,
    plot_sparkline,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="The Matchup Lab",
    page_icon="🔬",
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
        "roster_cache": {},
        "roster_team_ids": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# SQLite archetype persistence helpers
# ---------------------------------------------------------------------------

def _load_archetypes_from_db(g: MatchupGraph, season: str) -> None:
    """Restore off_archetype and def_role from SQLite into the in-memory graph."""
    try:
        from db import get_archetypes
        stored = get_archetypes(season)
        for pid, labels in stored.items():
            player = g.players.get(pid)
            if not player:
                continue
            off_val = labels.get("off_archetype")
            def_val = labels.get("def_role")
            if off_val:
                try:
                    player.off_archetype = OffensiveArchetype(off_val)
                except ValueError:
                    pass
            if def_val:
                try:
                    player.def_role = DefensiveRole(def_val)
                except ValueError:
                    pass
    except Exception:
        pass


def _save_archetypes_to_db(g: MatchupGraph, season: str) -> None:
    """Persist off_archetype and def_role from in-memory graph to SQLite."""
    try:
        from db import upsert_archetypes
        for pid, player in g.players.items():
            off = player.off_archetype.value if player.off_archetype else None
            def_ = player.def_role.value if player.def_role else None
            upsert_archetypes(pid, off, def_, season)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Auto-load from cache on first render (no button click required)
# ---------------------------------------------------------------------------

if not st.session_state.data_loaded and not st.session_state.get("_autoload_done"):
    st.session_state._autoload_done = True
    _auto_season = st.session_state.get("season", "2025-26")
    _auto_csv = CACHE_DIR / f"matchups_{_auto_season.replace('-', '_')}_Regular_Season.csv"
    if _auto_csv.exists():
        try:
            _auto_df = load_matchup_data(
                _auto_season, "Regular Season",
                min_possessions=st.session_state.get("min_poss", 20),
            )
            if not _auto_df.empty:
                _auto_g = MatchupGraph()
                _auto_g.build_from_dataframe(
                    _auto_df, min_possessions=st.session_state.get("min_poss", 20)
                )
                st.session_state.graph = _auto_g
                st.session_state.data_loaded = True
                st.session_state.season = _auto_season
                enrich_graph(_auto_g, season=_auto_season)
                st.session_state.enriched = True

                # Load archetypes from SQLite; compute + save if none stored yet
                _load_archetypes_from_db(_auto_g, _auto_season)
                _arch_count = sum(
                    1 for p in _auto_g.players.values() if p.off_archetype is not None
                )
                if _arch_count == 0:
                    try:
                        classify_archetypes_from_existing_stats(_auto_g)
                        _save_archetypes_to_db(_auto_g, _auto_season)
                    except Exception:
                        pass

                st.session_state.team_data_loaded = False
        except Exception:
            pass

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


def _get_career_df_fast(player_id: int):
    """
    Return (career_df, weighted_baseline) from file cache without live API calls.
    Returns (None, {}) if no cache exists so callers fall back gracefully.
    """
    proc_cache = CACHE_DIR / f"career_splits_processed_{player_id}.json"
    raw_cache  = CACHE_DIR / f"career_splits_{player_id}.json"
    if proc_cache.exists() or raw_cache.exists():
        try:
            return get_player_career_splits(player_id)
        except Exception:
            pass
    return None, {}



# ---------------------------------------------------------------------------
# Team data loader — defined here so sidebar and tab can both call it
# ---------------------------------------------------------------------------

def _do_load_team_data(season_end_year: int, force: bool = False) -> bool:
    """Fetch team stats, standings, and playoff bracket from BBRef and cache boxscores.
    Returns True on success."""
    import datetime
    try:
        tdf = get_bbref_team_stats(season_end_year, force=force)
        bracket = get_bbref_playoff_bracket(season_end_year, force=force)
        if not tdf.empty:
            tdf["FULL_NAME"] = tdf["TEAM_NAME"]
            tdf["TeamName"] = tdf["TEAM_NAME"]
        st.session_state.team_stats_df = tdf if not tdf.empty else None
        st.session_state.standings_df = tdf if not tdf.empty else None
        st.session_state.playoff_bracket_list = bracket
        st.session_state.playoff_series_df = None
        st.session_state.team_data_loaded = True
        st.session_state.team_data_updated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if force:
            st.session_state.roster_cache = {}
        if bracket:
            prefetch_playoff_boxscores(bracket, season_end_year)
        return True
    except Exception:
        return False


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

    enrich_btn = st.button("🔄 Refresh Data", use_container_width=True,
                           type="primary",
                           help="Re-fetch latest player stats and game logs from NBA API",
                           disabled=not st.session_state.data_loaded)
    st.caption("Data loads automatically from cache on startup.")

    # Refresh Data — re-fetches from NBA API, re-enriches, re-classifies archetypes
    if enrich_btn and st.session_state.graph:
        _enrich_season = st.session_state.get("season", "2025-26")
        _enrich_end    = int(_enrich_season.split("-")[0]) + 1

        prog_bar = st.progress(0, text="Refreshing player stats…")

        def _prog(i, tot, name):
            prog_bar.progress(i / max(tot, 1), text=f"Refreshing {name}… ({i}/{tot})")

        with st.spinner("Fetching latest bio + stats from NBA API…"):
            try:
                enrich_graph(
                    st.session_state.graph,
                    season=_enrich_season,
                    progress_callback=_prog,
                    force_refresh_epm=True,
                )
                st.session_state.enriched = True
                graph = st.session_state.graph
            except Exception as _e:
                st.error(f"Stat refresh error: {_e}")
            finally:
                prog_bar.empty()

        with st.spinner("Refreshing team stats and bracket…"):
            _do_load_team_data(_enrich_end)

        if _BBREF_AVAILABLE and st.session_state.graph:
            _all_names = list({p.name for p in st.session_state.graph.players.values()})
            prog_bar2 = st.progress(0, text="Pre-fetching game logs…")
            for _li, _lname in enumerate(_all_names):
                prog_bar2.progress((_li + 1) / max(len(_all_names), 1),
                                   text=f"Fetching logs: {_lname}")
                try:
                    get_current_season_logs(_lname, _enrich_end)
                    get_playoff_logs(_lname, _enrich_end)
                except Exception:
                    pass
            prog_bar2.empty()

        with st.spinner("Re-computing archetypes…"):
            try:
                _rsum = classify_archetypes_from_existing_stats(st.session_state.graph)
                _save_archetypes_to_db(st.session_state.graph, _enrich_season)
                graph = st.session_state.graph
            except Exception:
                pass

        st.success("Data refreshed — stats, game logs, and archetypes updated.")

    # Graph summary in sidebar
    if st.session_state.data_loaded and st.session_state.graph:
        g = st.session_state.graph
        summ = g.get_summary()
        st.markdown("---")
        st.markdown("### Graph Summary")
        st.metric("Players (Offense)", summ["offensive_players"])
        st.metric("Players (Defense)", summ["defensive_players"])
        st.metric("Matchup Edges", summ["total_edges"])
        st.metric("Avg PPP", f"{summ['avg_ppp']:.3f}")
        _off_arch_n = sum(1 for p in g.players.values() if p.off_archetype)
        _def_role_n = sum(1 for p in g.players.values() if p.def_role)
        if _off_arch_n:
            st.caption(f"✅ {_off_arch_n} off archetypes · {_def_role_n} def roles")
        else:
            st.caption("Archetypes computing on next render…")

    graph = st.session_state.graph


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

# Hero header
st.markdown(
    """
    <div style="text-align:center; padding: 24px 0 8px 0;">
        <h1 style="font-size:2.4rem; font-weight:800; color:#FAFAFA; margin:0;">
            🔬 The Matchup Lab
        </h1>
        <p style="color:#9CA3AF; font-size:1.05rem; margin:6px 0 0 0;">
            Archetype-driven investigation of NBA offensive–defensive matchups
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
# Auto-load team data on first run of the session
# ---------------------------------------------------------------------------
if not st.session_state.get("team_data_loaded"):
    _auto_season_end = int(st.session_state.get("season", "2025-26").split("-")[0]) + 1
    with st.spinner("Loading team stats and playoff bracket…"):
        _do_load_team_data(_auto_season_end)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬  Matchup Lab",
    "👤  Player Report",
    "🔄  Find Comparable Players",
    "🗂  Archetype Browser",
    "🤖  Full Analysis",
])


# ===========================================================================
# TAB 1 — Matchup Lookup
# ===========================================================================
with tab1:
    st.markdown('<div class="section-header">Matchup Lab</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-header">Player Report</div>', unsafe_allow_html=True)
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
# TAB 3 — Find Comparable Players
# ===========================================================================
with tab3:
    st.markdown('<div class="section-header">Find Comparable Players</div>', unsafe_allow_html=True)
    st.markdown(
        "Find players with the most similar offensive or defensive profiles. "
        "Useful for trade evaluation, replacement scouting, and understanding which players "
        "represent the same archetype."
    )

    if _playoff_no_matchups:
        st.markdown(_PLAYOFF_MATCHUP_WARNING, unsafe_allow_html=True)

    fc_role = st.radio("Find comparable", ["Scorer", "Defender"], horizontal=True, key="fc_role")

    if fc_role == "Scorer":
        fc_names = graph.all_player_names("offense") if not _playoff_no_matchups else []
        fc_sel = st.selectbox("Select a Scorer", fc_names, key="fc_scorer")
    else:
        fc_names = graph.all_player_names("defense") if not _playoff_no_matchups else []
        fc_sel = st.selectbox("Select a Defender", fc_names, key="fc_defender")

    fc_slider_n = st.slider("Players to display in table", 3, 10, 5, key="fc_slider")
    fc_btn = st.button("Find Comparable Players", type="primary")

    if fc_btn and fc_sel:
        with st.spinner(f"Computing similarity for {fc_sel}…"):
            if fc_role == "Scorer":
                similar_fc = graph.find_similar_scorers(fc_sel, top_n=fc_slider_n)
            else:
                similar_fc = graph.find_similar_defenders(fc_sel, top_n=fc_slider_n)

        fc_pid = graph.find_player_id(fc_sel)
        fc_player = graph.players.get(fc_pid) if fc_pid else None

        if not similar_fc:
            st.warning(
                "Not enough shared opponents to compute similarity. "
                "Try a player with more matchup data or lower the min possessions filter."
            )
        else:
            st.markdown("---")

            if fc_player:
                meta_parts = list(filter(None, [
                    fc_player.position, fc_player.team, fc_player.height,
                    f"{fc_player.weight} lbs" if fc_player.weight else None,
                ]))
                arch_label = ""
                if fc_role == "Scorer" and fc_player.off_archetype:
                    arch_label = f" · {fc_player.off_archetype.value}"
                elif fc_role == "Defender" and fc_player.def_role:
                    arch_label = f" · {fc_player.def_role.value}"
                st.markdown(
                    f'<div class="player-badge">{fc_sel}</div> '
                    f'<span style="color:#9CA3AF; font-size:0.9rem;">'
                    f'{" · ".join(meta_parts)}{arch_label}</span>',
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="section-header">Similarity Rankings</div>', unsafe_allow_html=True)

            sim_table_rows = []
            for rank, s in enumerate(similar_fc[:fc_slider_n], 1):
                if fc_role == "Scorer":
                    sim_table_rows.append({
                        "Rank": rank,
                        "Player": s.get("scorer", "—"),
                        "Team": s.get("team") or "—",
                        "Position": s.get("position") or "—",
                        "Archetype": s.get("archetype") or "—",
                        "Combined MPS": f"{s['combined_score']:.3f}",
                        "MPS_off": f"{s.get('mps_off', s['combined_score']):.3f}",
                        "Shared Opp.": s.get("shared_opponents", 0),
                        "Avg PPP": f"{s['avg_ppp_off']:.3f}" if s.get("avg_ppp_off") else "—",
                    })
                else:
                    sim_table_rows.append({
                        "Rank": rank,
                        "Player": s.get("defender", "—"),
                        "Team": s.get("team") or "—",
                        "Position": s.get("position") or "—",
                        "Archetype": s.get("archetype") or "—",
                        "Combined MPS": f"{s['combined_score']:.3f}",
                        "MPS_def": f"{s.get('mps_def', s['combined_score']):.3f}",
                        "Shared Opp.": s.get("shared_opponents", 0),
                        "Avg PPP Allowed": f"{s['avg_ppp_def']:.3f}" if s.get("avg_ppp_def") else "—",
                    })

            st.dataframe(pd.DataFrame(sim_table_rows), hide_index=True, use_container_width=True)

            # Horizontal bar chart of combined MPS scores
            import plotly.graph_objects as go
            _fc_player_labels = [r["Player"] for r in sim_table_rows]
            _fc_scores = [float(r["Combined MPS"]) for r in sim_table_rows]
            _fc_fig = go.Figure(go.Bar(
                x=_fc_scores,
                y=_fc_player_labels,
                orientation="h",
                marker_color="#3b82f6",
                text=[f"{s:.3f}" for s in _fc_scores],
                textposition="outside",
            ))
            _fc_fig.update_layout(
                title=f"Most Similar {'Scorers' if fc_role == 'Scorer' else 'Defenders'} to {fc_sel}",
                xaxis_title="Combined MPS",
                yaxis={"categoryorder": "total ascending"},
                plot_bgcolor="#0a0e17", paper_bgcolor="#0a0e17",
                font_color="#f1f5f9",
                height=max(250, 60 * len(_fc_player_labels)),
                margin={"l": 10, "r": 80, "t": 40, "b": 40},
            )
            st.plotly_chart(_fc_fig, use_container_width=True)

            st.markdown("---")

            if st.session_state.api_key:
                if st.button("Generate Similarity Report", type="primary", key="fc_report_btn"):
                    with st.spinner("Generating scouting report…"):
                        _fc_report = generate_similarity_report(
                            fc_player, similar_fc[:5], graph, st.session_state.api_key,
                            role="offense" if fc_role == "Scorer" else "defense",
                        )
                    st.markdown(f"### Scouting Report: {fc_sel} — Comparable Players")
                    st.markdown(f'<div class="report-box">{_fc_report}</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="info-box">Enter an Anthropic API key in the sidebar '
                    'to generate a similarity report.</div>',
                    unsafe_allow_html=True,
                )

            with st.expander("How is similarity calculated?"):
                if fc_role == "Scorer":
                    st.markdown("""
                    **MPS_off (Matchup Profile Score — Offense)**

                    | Component | Weight | Meaning |
                    |---|---|---|
                    | Jaccard (shared defenders faced) | 0.20 | Overlap of defensive opponents |
                    | PPP delta correlation | 0.15 | How similarly they score vs shared defenders |
                    | Shot zone similarity | 0.15 | Zone distribution alignment |
                    | Usage archetype sim | 0.15 | USG%, AST%, and role similarity |
                    | Offensive tier sim | 0.35 | Weighted cosine of z-scored stat vectors |

                    Only pairs with ≥ 3 shared opponents are considered.
                    """)
                else:
                    st.markdown("""
                    **MPS_def (Matchup Profile Score — Defense)**

                    | Component | Weight | Meaning |
                    |---|---|---|
                    | Jaccard (shared offensive opponents) | 0.20 | Overlap of opponents guarded |
                    | PPP delta correlation | 0.15 | How similarly they allow vs shared opponents |
                    | Shot profile similarity | 0.15 | Rim/mid/3 allowed profile alignment |
                    | Physical archetype sim | 0.15 | Height, weight, and position similarity |
                    | Defensive tier sim | 0.35 | Weighted cosine of z-scored stat vectors |

                    Only pairs with ≥ 3 shared opponents are considered.
                    """)


# ===========================================================================
# TAB 4 — Archetype Browser
# ===========================================================================
with tab4:
    st.markdown('<div class="section-header">Archetype Browser</div>', unsafe_allow_html=True)
    st.markdown(
        "Browse players by offensive archetype or defensive role. "
        "All classifications emerge from statistics — no player names are hardcoded."
    )

    _ab_left, _ab_right = st.columns(2)

    with _ab_left:
        st.markdown("#### Offensive Archetypes")
        _off_arch_sel = st.selectbox(
            "Select an offensive archetype",
            [a.value for a in OffensiveArchetype],
            key="ab_off_arch",
        )
        if _off_arch_sel:
            _sel_arch = OffensiveArchetype(_off_arch_sel)
            _arch_players = sorted(
                [p for p in graph.players.values() if p.off_archetype == _sel_arch],
                key=lambda p: p.ppg or 0, reverse=True,
            )
            if _arch_players:
                st.caption(f"{len(_arch_players)} players classified as {_off_arch_sel}")
                _arch_rows = []
                for _ap in _arch_players:
                    _r = {"Player": _ap.name, "Team": _ap.team or "—",
                          "Position": _ap.position or "—", "Height": _ap.height or "—"}
                    if _sel_arch in (OffensiveArchetype.PRIMARY_BH, OffensiveArchetype.SECONDARY_BH):
                        _r.update({"PPG": f"{_ap.ppg:.1f}" if _ap.ppg else "—",
                                   "AST%": f"{_ap.ast_pct:.1%}" if _ap.ast_pct else "—",
                                   "USG%": f"{_ap.usg_pct:.1%}" if _ap.usg_pct else "—"})
                    elif _sel_arch == OffensiveArchetype.SLASHER:
                        _r.update({"Rim FGA/100": f"{_ap.p_fga_rim_100:.1f}" if _ap.p_fga_rim_100 else "—",
                                   "Rim FG%": f"{_ap.p_fgpct_rim:.1%}" if _ap.p_fgpct_rim else "—"})
                    elif _sel_arch in (OffensiveArchetype.OFF_SCREEN_SHOOTER,
                                       OffensiveArchetype.MOVEMENT_SHOOTER,
                                       OffensiveArchetype.STATIONARY_SHOOTER):
                        _r.update({"3PA/100": f"{_ap.p_fg3a_100:.1f}" if _ap.p_fg3a_100 else "—",
                                   "3P%": f"{_ap.fg3_pct:.1%}" if _ap.fg3_pct else "—",
                                   "TS%": f"{_ap.ts_pct:.1%}" if _ap.ts_pct else "—"})
                    elif _sel_arch == OffensiveArchetype.ATHLETIC_FINISHER:
                        _r.update({"ORB/100": f"{_ap.p_orb_100:.1f}" if _ap.p_orb_100 else "—",
                                   "Rim FGA/100": f"{_ap.p_fga_rim_100:.1f}" if _ap.p_fga_rim_100 else "—"})
                    else:
                        _r.update({"PPG": f"{_ap.ppg:.1f}" if _ap.ppg else "—",
                                   "USG%": f"{_ap.usg_pct:.1%}" if _ap.usg_pct else "—",
                                   "3P%": f"{_ap.fg3_pct:.1%}" if _ap.fg3_pct else "—"})
                    _arch_rows.append(_r)
                st.dataframe(pd.DataFrame(_arch_rows), hide_index=True, use_container_width=True)
            else:
                st.markdown(
                    '<div class="info-box">'
                    f'No players are currently classified as <b>{_off_arch_sel}</b>. '
                    'Offensive archetype classification requires Synergy play-type data '
                    '(pnr_bh_freq, iso_freq, drives_per75, etc.). '
                    'Classifications will populate once that data pipeline is connected.'
                    '</div>',
                    unsafe_allow_html=True,
                )

    with _ab_right:
        st.markdown("#### Defensive Roles")
        _def_role_sel = st.selectbox(
            "Select a defensive role",
            [r.value for r in DefensiveRole],
            key="ab_def_role",
        )
        if _def_role_sel:
            _sel_role = DefensiveRole(_def_role_sel)
            _role_players = sorted(
                [p for p in graph.players.values() if p.def_role == _sel_role],
                key=lambda p: p.epm_def or 0, reverse=True,
            )
            if _role_players:
                st.caption(f"{len(_role_players)} players classified as {_def_role_sel}")
                _role_rows = []
                for _rp in _role_players:
                    _r = {"Player": _rp.name, "Team": _rp.team or "—",
                          "Position": _rp.position or "—", "Height": _rp.height or "—"}
                    if _sel_role == DefensiveRole.POINT_OF_ATTACK:
                        _r.update({"DEPM": f"{_rp.epm_def:.2f}" if _rp.epm_def else "—",
                                   "STL/100": f"{_rp.p_stl_100:.1f}" if _rp.p_stl_100 else "—",
                                   "PPP Allowed": f"{_rp.avg_ppp_def:.3f}" if _rp.avg_ppp_def else "—"})
                    elif _sel_role == DefensiveRole.WING_STOPPER:
                        _r.update({"DEPM": f"{_rp.epm_def:.2f}" if _rp.epm_def else "—",
                                   "PPP Allowed": f"{_rp.avg_ppp_def:.3f}" if _rp.avg_ppp_def else "—"})
                    elif _sel_role == DefensiveRole.CHASER:
                        _r.update({"STL/100": f"{_rp.p_stl_100:.1f}" if _rp.p_stl_100 else "—",
                                   "DEPM": f"{_rp.epm_def:.2f}" if _rp.epm_def else "—"})
                    elif _sel_role == DefensiveRole.HELPER:
                        _r.update({"BLK/100": f"{_rp.p_blk_100:.1f}" if _rp.p_blk_100 else "—",
                                   "DEPM": f"{_rp.epm_def:.2f}" if _rp.epm_def else "—"})
                    elif _sel_role in (DefensiveRole.ANCHOR_BIG, DefensiveRole.MOBILE_BIG):
                        _r.update({"BLK/100": f"{_rp.p_blk_100:.1f}" if _rp.p_blk_100 else "—",
                                   "DRB/100": f"{_rp.p_drb_100:.1f}" if _rp.p_drb_100 else "—",
                                   "DEPM": f"{_rp.epm_def:.2f}" if _rp.epm_def else "—"})
                    else:
                        _r.update({"PPP Allowed": f"{_rp.avg_ppp_def:.3f}" if _rp.avg_ppp_def else "—",
                                   "DEPM": f"{_rp.epm_def:.2f}" if _rp.epm_def else "—"})
                    _role_rows.append(_r)
                st.dataframe(pd.DataFrame(_role_rows), hide_index=True, use_container_width=True)
            else:
                st.markdown(
                    '<div class="info-box">'
                    f'No players are currently classified as <b>{_def_role_sel}</b>. '
                    'Defensive role classification requires matchup assignment data '
                    '(pct_time_vs_pg, matchup_difficulty, def_positional_versatility, etc.). '
                    'Classifications will populate once that data pipeline is connected.'
                    '</div>',
                    unsafe_allow_html=True,
                )


# ===========================================================================
# TAB 5 — Full Analysis
# ===========================================================================
with tab5:
    st.markdown('<div class="section-header">Full Analysis</div>', unsafe_allow_html=True)
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
        ["Matchup Report", "Player Profile Report", "Defensive Similarity Report", "Ask the Analyst"],
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
                        _zones = {}
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
                            _career_parts.append(fmt_career_context(_cdf, _pname, _zones))
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

                        # Regular season game results — final scores only, from cache
                        try:
                            _rs_results = get_team_season_results(_team, _season_end)
                            _rs_ctx = fmt_team_season_results(_team, _rs_results)
                            if _rs_ctx:
                                _career_parts.append(_rs_ctx)
                        except Exception:
                            pass

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

                    # Playoff bracket + game box scores
                    if "playoffs" in _detected_concepts or _detected_teams:
                        _season_end_yr = int(st.session_state.get("season", "2025-26").split("-")[0]) + 1
                        # Use session-state bracket (kept fresh by Load Data / Refresh Data).
                        # Only hit BBRef live when session state is genuinely empty — hitting it
                        # on every question causes rate-limiting and drops the bracket entirely.
                        _bracket = st.session_state.get("playoff_bracket_list") or []
                        if not _bracket:
                            try:
                                _bracket = get_bbref_playoff_bracket(_season_end_yr)
                                if _bracket:
                                    st.session_state.playoff_bracket_list = _bracket
                            except Exception:
                                _bracket = []

                        _bracket_covered_teams = []

                        if _bracket:
                            _bracket_ctx = fmt_playoff_context(
                                _bracket,
                                filter_teams=_detected_teams if _detected_teams else None,
                            )
                            if _bracket_ctx:
                                _career_parts.append(_bracket_ctx)

                            # Full box scores for series matching detected teams
                            if _detected_teams:
                                for _s in _bracket:
                                    t1, t2 = _s["team1"].lower(), _s["team2"].lower()
                                    if any(
                                        ft.lower() in t1 or ft.lower() in t2
                                        for ft in _detected_teams
                                    ) and _s.get("games"):
                                        try:
                                            _box_ctx = get_playoff_series_boxscores(_s, _season_end_yr)
                                            if _box_ctx:
                                                _career_parts.append(_box_ctx)
                                                _bracket_covered_teams.extend([t1, t2])
                                        except Exception:
                                            pass

                        # Cache fallback — runs regardless of bracket state so cached
                        # game files are always injected even when BBRef is unavailable
                        if len(_detected_teams) >= 2:
                            _already_covered = all(
                                any(ft.lower() in ct for ct in _bracket_covered_teams)
                                for ft in _detected_teams[:2]
                            )
                            if not _already_covered:
                                try:
                                    _cache_ctx = get_cached_series_boxscores_for_teams(
                                        _detected_teams[:2]
                                    )
                                    if _cache_ctx:
                                        _career_parts.append(_cache_ctx)
                                except Exception:
                                    pass

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
                    )
                st.markdown("---")
                st.markdown("### The Analyst")
                st.markdown(f'<div class="report-box">{_analyst_report}</div>', unsafe_allow_html=True)
            else:
                st.warning("Enter a question first.")


