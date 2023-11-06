import logging

import duckdb
import pandas as pd

from src.constants import (
    DELTA_FILES,
    LAG_FILES,
    PATH_CLASE_BINARIA,
    PATH_CLASE_TERNARIA,
    TEST_MONTH,
    TRAINING_MONTHS,
    VALIDATION_MONTHS,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extract(con: duckdb.DuckDBPyConnection, path_parquet: str) -> None:
    logger.info("Extracting from %s", path_parquet)
    con.sql(
        f"""
        CREATE OR REPLACE TABLE competencia_03 AS (
            SELECT
                *
            FROM read_parquet('{path_parquet}')
        );
        """
    )


def transform(con: duckdb.DuckDBPyConnection, lags: bool = True, delta_lags: bool = False) -> None:
    in_clause_all = ", ".join([str(i) for i in TRAINING_MONTHS + VALIDATION_MONTHS + TEST_MONTH])
    logger.info("Creating ranks")
    con.sql(
        """
            CREATE OR REPLACE TABLE competencia_03 AS (
                SELECT
                    *,
                    RANK() OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes DESC) AS rank_foto_mes,
                FROM competencia_03
            );
            """
    )

    con.sql(
        """
            CREATE OR REPLACE TABLE competencia_03 AS (
                SELECT
                    *,
                    rank_foto_mes*-1 + 1 AS rank_foto_mes_2
                FROM competencia_03
            );
            """
    )
    logger.info("Creating ternaria")
    con.sql(
        """
        CREATE OR REPLACE TABLE competencia_03 AS (
            SELECT
                *,
                CASE
                    WHEN rank_foto_mes_2 = 0 THEN 'BAJA+2'
                    WHEN rank_foto_mes_2 =-1 THEN 'BAJA+1'
                    ELSE 'CONTINUA'
                END AS clase_ternaria
            FROM competencia_03
        );
        """
    )

    logger.info("Drop ternaria")
    con.sql(
        """
        ALTER TABLE competencia_03 DROP COLUMN rank_foto_mes;
        ALTER TABLE competencia_03 DROP COLUMN rank_foto_mes_2;
        """
    )

    if lags:
        logger.info("Creating lags")
        for i in LAG_FILES:
            with open(i) as f:
                query = f.read()
            logger.info("Creating lag %s", i)
            con.sql(
                f"""
                    CREATE OR REPLACE TABLE competencia_03 AS (
                        {query}
                    );
                """
            )
    if delta_lags:
        logger.info("Creating delta-lags")
        for i in DELTA_FILES:
            with open(i) as f:
                query = f.read()
            logger.info("Creating deta-lag %s", i)
            con.sql(
                f"""
                    CREATE OR REPLACE TABLE competencia_03 AS (
                        {query}
                    );
                    """
            )

    logger.info("Export terciaria %s", PATH_CLASE_TERNARIA)
    con.sql(
        f"""
        COPY competencia_03
        TO '{PATH_CLASE_TERNARIA}' (FORMAT PARQUET);
        """
    )

    logger.info("Create binaria")
    con.sql(
        f"""
            CREATE OR REPLACE TABLE competencia_03 AS (
                SELECT
                    *,
                    CASE
                        WHEN clase_ternaria = 'BAJA+2' THEN 1
                        WHEN clase_ternaria ='BAJA+1' THEN 1
                        WHEN clase_ternaria = 'CONTINUA' THEN 0
                        ELSE 0
                    END AS clase_binaria
                FROM competencia_03
                WHERE foto_mes IN ({in_clause_all})
            );
            """
    )

    con.sql(
        """
        ALTER TABLE competencia_03 DROP COLUMN clase_ternaria;
        """
    )
    logger.info("Export binaria %s", PATH_CLASE_BINARIA)
    con.sql(
        f"""
                COPY competencia_03
                TO '{PATH_CLASE_BINARIA}' (FORMAT PARQUET);
                """
    )


def get_dataframe(con: duckdb.DuckDBPyConnection, query: str) -> pd.DataFrame:
    logger.info("Querying and converting to dataframe")
    return con.sql(query).to_df()


def generate_small_dataset(con: duckdb.DuckDBPyConnection) -> None:
    con.sql(
        """
        CREATE OR REPLACE TABLE competencia_03_small AS (
        SELECT
            *
        FROM competencia_03
        WHERE numero_de_cliente IN (
            64852360,
            86372551,
            94680261,
            104451733,
            68015042,
            153977335,
            115341090,
            115297751,
            142553450,
            34002800,
            37929113,
            53867657,
            55545254,
            66563672,
            81769310,
            36842056
            )
        ORDER BY numero_de_cliente, foto_mes
        );
        """
    )

    con.sql(
        """
        COPY competencia_03_small
        TO 'datasets/raw/competencia_03_small.csv' (FORMAT CSV, HEADER);
        """
    )

    con.sql(
        """
        COPY competencia_03_small
        TO 'datasets/raw/competencia_03_small.parquet' (FORMAT PARQUET);
        """
    )
