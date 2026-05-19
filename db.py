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


def init_db() -> None:
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS matchup_cache (
                off_player_id   INTEGER NOT NULL,
                def_player_id   INTEGER NOT NULL,
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
                off_archetype   TEXT,
                def_role        TEXT,
                updated_at      TEXT,
                PRIMARY KEY (player_id, season)
            );

            CREATE TABLE IF NOT EXISTS synergy_playtypes (
                player_id           INTEGER NOT NULL,
                season              TEXT    NOT NULL,
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
                player_id       INTEGER NOT NULL,
                season          TEXT    NOT NULL,
                drives_pg       REAL,
                paint_touches_pg REAL,
                updated_at      TEXT,
                PRIMARY KEY (player_id, season)
            );
        """)


def upsert_matchups(df: pd.DataFrame, season: str, season_type: str) -> None:
    rows = [
        (
            int(row["OFF_PLAYER_ID"]),
            int(row["DEF_PLAYER_ID"]),
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
                off_player_id, def_player_id, season, season_type,
                games_played, possessions, points,
                fgm, fga, fg_pct,
                fg3m, fg3a, fg3_pct,
                assists, turnovers, blocks
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
) -> None:
    updated_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO player_archetypes
                (player_id, season, off_archetype, def_role, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (player_id, season, off_archetype, def_role, updated_at),
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
                player_id, season, scoring_possessions,
                pnr_bh_freq, iso_freq, post_freq, roll_freq, spot_freq,
                off_screen_freq, handoff_freq, cut_freq, putback_freq,
                updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )


def get_synergy_playtypes(season: str) -> Dict[int, Dict]:
    """Return Synergy play-type frequencies keyed by player_id."""
    with _connect() as conn:
        cursor = conn.execute(
            """
            SELECT player_id, scoring_possessions,
                   pnr_bh_freq, iso_freq, post_freq, roll_freq, spot_freq,
                   off_screen_freq, handoff_freq, cut_freq, putback_freq
            FROM synergy_playtypes WHERE season = ?
            """,
            (season,),
        )
        return {
            row["player_id"]: {
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
        (pid, season, d.get("drives_pg"), d.get("paint_touches_pg"), updated_at)
        for pid, d in data.items()
    ]
    with _connect() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO player_tracking
                (player_id, season, drives_pg, paint_touches_pg, updated_at)
            VALUES (?,?,?,?,?)
            """,
            rows,
        )


def get_player_tracking(season: str) -> Dict[int, Dict]:
    """Return drives_pg and paint_touches_pg keyed by player_id."""
    with _connect() as conn:
        cursor = conn.execute(
            "SELECT player_id, drives_pg, paint_touches_pg FROM player_tracking WHERE season = ?",
            (season,),
        )
        return {
            row["player_id"]: {
                "drives_pg":        row["drives_pg"],
                "paint_touches_pg": row["paint_touches_pg"],
            }
            for row in cursor.fetchall()
        }
