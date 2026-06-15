"""
SQLite caching layer for the NBA Matchup Project.

Additive on top of data_loader.py — does not replace any existing logic.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from models import Player

DB_PATH = Path("data/cache/matchup_lab.db")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _add_column_if_missing(conn: sqlite3.Connection, table: str, col_def: str) -> None:
    """ALTER TABLE … ADD COLUMN, silently ignoring 'duplicate column name' errors."""
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
    except sqlite3.OperationalError:
        pass  # column already exists


def init_db() -> None:
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS matchup_cache (
                off_player_id   INTEGER NOT NULL,
                off_player_name TEXT,
                def_player_id   INTEGER NOT NULL,
                def_player_name TEXT,
                season          TEXT    NOT NULL,
                season_type     TEXT    NOT NULL,
                games_played    INTEGER,
                possessions     REAL,
                points          REAL,
                fgm             REAL,
                fga             REAL,
                fg_pct          REAL,
                fg3m            REAL,
                fg3a            REAL,
                fg3_pct         REAL,
                assists         REAL,
                turnovers       REAL,
                blocks          REAL,
                PRIMARY KEY (off_player_id, def_player_id, season, season_type)
            );

            CREATE TABLE IF NOT EXISTS player_bio (
                player_id   INTEGER PRIMARY KEY,
                name        TEXT,
                position    TEXT,
                team        TEXT,
                height      TEXT,
                weight      INTEGER,
                ppg         REAL,
                rpg         REAL,
                apg         REAL,
                spg         REAL,
                bpg         REAL,
                tov         REAL,
                mpg         REAL,
                fg_pct      REAL,
                fg3_pct     REAL,
                ft_pct      REAL,
                ts_pct      REAL,
                games       INTEGER,
                off_rating  REAL,
                def_rating  REAL,
                net_rating  REAL,
                usg_pct     REAL,
                pie         REAL,
                ast_pct     REAL,
                epm_off     REAL,
                epm_def     REAL,
                epm_tot     REAL,
                avg_ppp_off REAL,
                avg_ppp_def REAL
            );

            CREATE TABLE IF NOT EXISTS player_archetypes (
                player_id       INTEGER NOT NULL,
                season          TEXT    NOT NULL,
                name            TEXT,
                off_archetype   TEXT,
                def_role        TEXT,
                updated_at      TEXT,
                PRIMARY KEY (player_id, season)
            );

            CREATE TABLE IF NOT EXISTS synergy_playtypes (
                player_id           INTEGER NOT NULL,
                season              TEXT    NOT NULL,
                name                TEXT,
                scoring_possessions REAL,
                pnr_bh_freq         REAL,
                iso_freq            REAL,
                post_freq           REAL,
                roll_freq           REAL,
                spot_freq           REAL,
                off_screen_freq     REAL,
                handoff_freq        REAL,
                cut_freq            REAL,
                putback_freq        REAL,
                updated_at          TEXT,
                PRIMARY KEY (player_id, season)
            );

            CREATE TABLE IF NOT EXISTS player_tracking (
                player_id        INTEGER NOT NULL,
                season           TEXT    NOT NULL,
                name             TEXT,
                drives_pg        REAL,
                paint_touches_pg REAL,
                updated_at       TEXT,
                PRIMARY KEY (player_id, season)
            );
        """)
        # Migrate existing DBs — add name columns if they don't exist yet
        _add_column_if_missing(conn, "matchup_cache",   "off_player_name TEXT")
        _add_column_if_missing(conn, "matchup_cache",   "def_player_name TEXT")
        _add_column_if_missing(conn, "player_archetypes", "name TEXT")
        _add_column_if_missing(conn, "synergy_playtypes", "name TEXT")
        _add_column_if_missing(conn, "player_tracking",   "name TEXT")


def upsert_matchups(df: pd.DataFrame, season: str, season_type: str) -> None:
    rows = [
        (
            int(row["OFF_PLAYER_ID"]),
            row.get("OFF_PLAYER_NAME"),
            int(row["DEF_PLAYER_ID"]),
            row.get("DEF_PLAYER_NAME"),
            season,
            season_type,
            row.get("GP"),
            row.get("PARTIAL_POSS"),
            row.get("PLAYER_PTS"),
            row.get("MATCHUP_FGM"),
            row.get("MATCHUP_FGA"),
            row.get("MATCHUP_FG_PCT"),
            row.get("MATCHUP_FG3M"),
            row.get("MATCHUP_FG3A"),
            row.get("MATCHUP_FG3_PCT"),
            row.get("MATCHUP_AST"),
            row.get("MATCHUP_TOV"),
            row.get("MATCHUP_BLK"),
        )
        for _, row in df.iterrows()
    ]

    with _connect() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO matchup_cache (
                off_player_id, off_player_name, def_player_id, def_player_name,
                season, season_type,
                games_played, possessions, points,
                fgm, fga, fg_pct,
                fg3m, fg3a, fg3_pct,
                assists, turnovers, blocks
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )


def upsert_players(players_dict: Dict[int, Player]) -> None:
    rows = [
        (
            p.player_id,
            p.name,
            p.position,
            p.team,
            p.height,
            p.weight,
            p.ppg,
            p.rpg,
            p.apg,
            p.spg,
            p.bpg,
            p.tov,
            p.mpg,
            p.fg_pct,
            p.fg3_pct,
            p.ft_pct,
            p.ts_pct,
            p.games,
            p.off_rating,
            p.def_rating,
            p.net_rating,
            p.usg_pct,
            p.pie,
            p.ast_pct,
            p.epm_off,
            p.epm_def,
            p.epm_tot,
            p.avg_ppp_off,
            p.avg_ppp_def,
        )
        for p in players_dict.values()
    ]

    with _connect() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO player_bio (
                player_id, name, position, team, height, weight,
                ppg, rpg, apg, spg, bpg, tov, mpg,
                fg_pct, fg3_pct, ft_pct, ts_pct, games,
                off_rating, def_rating, net_rating, usg_pct, pie, ast_pct,
                epm_off, epm_def, epm_tot,
                avg_ppp_off, avg_ppp_def
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )


def upsert_archetypes(
    player_id: int,
    off_archetype: Optional[str],
    def_role: Optional[str],
    season: str,
    name: Optional[str] = None,
) -> None:
    updated_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO player_archetypes
                (player_id, season, name, off_archetype, def_role, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (player_id, season, name, off_archetype, def_role, updated_at),
        )


def get_cached_matchups(season: str, season_type: str) -> pd.DataFrame:
    with _connect() as conn:
        cursor = conn.execute(
            """
            SELECT
                off_player_id   AS OFF_PLAYER_ID,
                def_player_id   AS DEF_PLAYER_ID,
                games_played    AS GP,
                possessions     AS PARTIAL_POSS,
                points          AS PLAYER_PTS,
                fgm             AS MATCHUP_FGM,
                fga             AS MATCHUP_FGA,
                fg_pct          AS MATCHUP_FG_PCT,
                fg3m            AS MATCHUP_FG3M,
                fg3a            AS MATCHUP_FG3A,
                fg3_pct         AS MATCHUP_FG3_PCT,
                assists         AS MATCHUP_AST,
                turnovers       AS MATCHUP_TOV,
                blocks          AS MATCHUP_BLK
            FROM matchup_cache
            WHERE season = ? AND season_type = ?
            """,
            (season, season_type),
        )
        rows = cursor.fetchall()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([dict(r) for r in rows])


def get_archetypes(season: str) -> Dict[int, Dict[str, Optional[str]]]:
    with _connect() as conn:
        cursor = conn.execute(
            "SELECT player_id, off_archetype, def_role FROM player_archetypes WHERE season = ?",
            (season,),
        )
        return {
            row["player_id"]: {
                "off_archetype": row["off_archetype"],
                "def_role": row["def_role"],
            }
            for row in cursor.fetchall()
        }


def upsert_synergy_playtypes(data: Dict[int, Dict], season: str) -> None:
    """Bulk upsert Synergy play-type frequency data keyed by player_id."""
    updated_at = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            pid, season,
            d.get("name"),
            d.get("scoring_possessions"),
            d.get("pnr_bh_freq"),
            d.get("iso_freq"),
            d.get("post_freq"),
            d.get("roll_freq"),
            d.get("spot_freq"),
            d.get("off_screen_freq"),
            d.get("handoff_freq"),
            d.get("cut_freq"),
            d.get("putback_freq"),
            updated_at,
        )
        for pid, d in data.items()
    ]
    with _connect() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO synergy_playtypes (
                player_id, season, name, scoring_possessions,
                pnr_bh_freq, iso_freq, post_freq, roll_freq, spot_freq,
                off_screen_freq, handoff_freq, cut_freq, putback_freq,
                updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )


def get_synergy_playtypes(season: str) -> Dict[int, Dict]:
    """Return Synergy play-type frequencies keyed by player_id."""
    with _connect() as conn:
        cursor = conn.execute(
            """
            SELECT player_id, name, scoring_possessions,
                   pnr_bh_freq, iso_freq, post_freq, roll_freq, spot_freq,
                   off_screen_freq, handoff_freq, cut_freq, putback_freq
            FROM synergy_playtypes WHERE season = ?
            """,
            (season,),
        )
        return {
            row["player_id"]: {
                "name":                row["name"],
                "scoring_possessions": row["scoring_possessions"],
                "pnr_bh_freq":         row["pnr_bh_freq"],
                "iso_freq":            row["iso_freq"],
                "post_freq":           row["post_freq"],
                "roll_freq":           row["roll_freq"],
                "spot_freq":           row["spot_freq"],
                "off_screen_freq":     row["off_screen_freq"],
                "handoff_freq":        row["handoff_freq"],
                "cut_freq":            row["cut_freq"],
                "putback_freq":        row["putback_freq"],
            }
            for row in cursor.fetchall()
        }


def upsert_player_tracking(data: Dict[int, Dict], season: str) -> None:
    """Bulk upsert drives and paint-touch tracking stats keyed by player_id."""
    updated_at = datetime.now(timezone.utc).isoformat()
    rows = [
        (pid, season, d.get("name"), d.get("drives_pg"), d.get("paint_touches_pg"), updated_at)
        for pid, d in data.items()
    ]
    with _connect() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO player_tracking
                (player_id, season, name, drives_pg, paint_touches_pg, updated_at)
            VALUES (?,?,?,?,?,?)
            """,
            rows,
        )


def get_player_tracking(season: str) -> Dict[int, Dict]:
    """Return drives_pg and paint_touches_pg keyed by player_id."""
    with _connect() as conn:
        cursor = conn.execute(
            "SELECT player_id, name, drives_pg, paint_touches_pg FROM player_tracking WHERE season = ?",
            (season,),
        )
        return {
            row["player_id"]: {
                "name":             row["name"],
                "drives_pg":        row["drives_pg"],
                "paint_touches_pg": row["paint_touches_pg"],
            }
            for row in cursor.fetchall()
        }


def export_players_csv(
    season: str = "2025-26",
    out_path: str = "NBA Players.csv",
) -> int:
    """
    Export all players in the tool to a CSV for manual archetype assignment.

    Columns: player_id, name, team, position, height, weight,
             ppg, rpg, apg, spg, bpg, tov, mpg, fg_pct, fg3_pct, ft_pct,
             ts_pct, games, off_rating, def_rating, net_rating,
             usg_pct, pie, ast_pct,
             epm_off, epm_def, epm_tot,
             p_pts_100, p_ast_100, p_blk_100, p_stl_100, p_drb_100,
             p_orb_100, p_tov_100, p_fga_rim_100, p_fga_mid_100, p_fg3a_100,
             scoring_possessions, pnr_bh_freq, iso_freq, post_freq,
             roll_freq, spot_freq, off_screen_freq, handoff_freq,
             cut_freq, putback_freq,
             drives_pg, paint_touches_pg,
             computed_off_archetype, computed_def_archetype,
             offensive_archetype, defensive_archetype

    Returns the number of rows written.
    """
    import json
    from pathlib import Path as _Path

    cache_dir = _Path("data/cache")

    # ── 1. Collect all players from matchup CSVs ──────────────────────────────
    players: Dict[int, Dict] = {}
    for csv_file in sorted(cache_dir.glob(f"matchups_*{season.replace('-','_')}*.csv")):
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue
        for _, row in df.iterrows():
            for id_col, name_col in [
                ("OFF_PLAYER_ID", "OFF_PLAYER_NAME"),
                ("DEF_PLAYER_ID", "DEF_PLAYER_NAME"),
            ]:
                try:
                    pid = int(row[id_col])
                    name = str(row[name_col])
                    if pid not in players:
                        players[pid] = {"name": name}
                except (KeyError, ValueError):
                    pass

    if not players:
        return 0

    # ── 2. Bio cache (position, team, height, weight) ─────────────────────────
    for pid in list(players.keys()):
        bio_file = cache_dir / f"bio_{pid}.json"
        if bio_file.exists():
            try:
                bio = json.loads(bio_file.read_text())
                players[pid].update({
                    "team":     bio.get("team"),
                    "position": bio.get("position"),
                    "height":   bio.get("height"),
                    "weight":   bio.get("weight"),
                })
            except Exception:
                pass

    # ── 3. Per-game stats cache ───────────────────────────────────────────────
    for pid in list(players.keys()):
        stats_file = cache_dir / f"stats_{pid}_{season.replace('-', '_')}.json"
        if stats_file.exists():
            try:
                sg = json.loads(stats_file.read_text())
                for k in ("ppg", "rpg", "apg", "spg", "bpg", "tov",
                          "mpg", "fg_pct", "fg3_pct", "ft_pct", "ts_pct", "games"):
                    players[pid][k] = sg.get(k)
            except Exception:
                pass

    # ── 4. EPM + per-100 stats ────────────────────────────────────────────────
    epm_file = cache_dir / "epm_2026.json"
    if epm_file.exists():
        try:
            epm_data = json.loads(epm_file.read_text())
            for pid, p in players.items():
                name_lower = (p.get("name") or "").lower()
                epm = epm_data.get(name_lower, {})
                if epm:
                    for k, col in [
                        ("off", "epm_off"), ("def", "epm_def"), ("tot", "epm_tot"),
                        ("p_pts_100", "p_pts_100"), ("p_ast_100", "p_ast_100"),
                        ("p_blk_100", "p_blk_100"), ("p_stl_100", "p_stl_100"),
                        ("p_drb_100", "p_drb_100"), ("p_orb_100", "p_orb_100"),
                        ("p_tov_100", "p_tov_100"), ("p_fga_rim_100", "p_fga_rim_100"),
                        ("p_fga_mid_100", "p_fga_mid_100"), ("p_fg3a_100", "p_fg3a_100"),
                    ]:
                        players[pid][col] = epm.get(k)
        except Exception:
            pass

    # ── 5. Synergy play-type freqs from DB ────────────────────────────────────
    synergy = get_synergy_playtypes(season)
    for pid, s in synergy.items():
        if pid in players:
            for k in ("scoring_possessions", "pnr_bh_freq", "iso_freq", "post_freq",
                      "roll_freq", "spot_freq", "off_screen_freq", "handoff_freq",
                      "cut_freq", "putback_freq"):
                players[pid][k] = s.get(k)

    # ── 6. Tracking from DB ───────────────────────────────────────────────────
    tracking = get_player_tracking(season)
    for pid, t in tracking.items():
        if pid in players:
            players[pid]["drives_pg"] = t.get("drives_pg")
            players[pid]["paint_touches_pg"] = t.get("paint_touches_pg")

    # ── 7. Computed archetypes from DB ────────────────────────────────────────
    archetypes = get_archetypes(season)
    for pid, arch in archetypes.items():
        if pid in players:
            players[pid]["computed_off_archetype"] = arch.get("off_archetype")
            players[pid]["computed_def_archetype"] = arch.get("def_role")

    # ── 8. Build DataFrame and export ────────────────────────────────────────
    columns = [
        "player_id", "name", "team", "position", "height", "weight",
        "ppg", "rpg", "apg", "spg", "bpg", "tov", "mpg",
        "fg_pct", "fg3_pct", "ft_pct", "ts_pct", "games",
        "off_rating", "def_rating", "net_rating",
        "usg_pct", "pie", "ast_pct",
        "epm_off", "epm_def", "epm_tot",
        "p_pts_100", "p_ast_100", "p_blk_100", "p_stl_100",
        "p_drb_100", "p_orb_100", "p_tov_100",
        "p_fga_rim_100", "p_fga_mid_100", "p_fg3a_100",
        "scoring_possessions",
        "pnr_bh_freq", "iso_freq", "post_freq", "roll_freq", "spot_freq",
        "off_screen_freq", "handoff_freq", "cut_freq", "putback_freq",
        "drives_pg", "paint_touches_pg",
        "computed_off_archetype", "computed_def_archetype",
        "offensive_archetype", "defensive_archetype",
    ]

    rows_out = []
    for pid, p in sorted(players.items(), key=lambda x: (x[1].get("name") or "")):
        row_data = {"player_id": pid}
        for col in columns[1:]:
            row_data[col] = p.get(col)
        # blank override columns for manual entry
        row_data["offensive_archetype"] = ""
        row_data["defensive_archetype"] = ""
        rows_out.append(row_data)

    df_out = pd.DataFrame(rows_out, columns=columns)
    df_out.to_csv(out_path, index=False)
    return len(df_out)
