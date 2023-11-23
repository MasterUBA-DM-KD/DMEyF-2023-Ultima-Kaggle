import logging

import duckdb

from src.constants import (
    DELTA_FILES,
    INFLATION,
    LAG_FILES,
    MOVING_AVG_FILES,
    PATH_INFLATION_FINAL,
    TEST_MONTH,
    TRAINING_MONTHS,
    TREND_FILES,
)
from src.preprocess.utils import create_feature, get_argentina_inflation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extract(con: duckdb.DuckDBPyConnection, path_parquet: str, where_clause: str = "1=1") -> None:
    logger.info("Extracting from %s", path_parquet)
    con.sql(
        f"""
        CREATE OR REPLACE TABLE competencia_03 AS (
            SELECT
                *
            FROM read_parquet('{path_parquet}')
            WHERE {where_clause}
        );
        """
    )


def transform(
    con: duckdb.DuckDBPyConnection,
    inflation: bool = True,
    lag: bool = True,
    delta_lag: bool = True,
    moving_avg: bool = True,
    trend: bool = True,
) -> None:
    logger.info("Creating targets")
    create_targets(con)
    if inflation:
        logger.info("Adjusting inflation")
        adjust_inflation(con)
    if lag:
        logger.info("Creating lags")
        create_feature(con, LAG_FILES)
    if delta_lag:
        logger.info("Creating delta lags")
        create_feature(con, DELTA_FILES)
    if moving_avg:
        logger.info("Creating moving averages")
        create_feature(con, MOVING_AVG_FILES)
    if trend:
        logger.info("Creating trend")
        create_feature(con, TREND_FILES)


def load(con: duckdb.DuckDBPyConnection, path_final: str, where_clause: str = "1=1") -> None:
    in_clause_all = ", ".join([str(i) for i in TRAINING_MONTHS + TEST_MONTH])
    logger.info("Filter dataset to training and test months")
    con.sql(
        f"""
        CREATE OR REPLACE TABLE competencia_03 AS (
            SELECT
                *
            FROM competencia_03
            WHERE {where_clause}
            -- foto_mes IN ({in_clause_all})
        );
        """
    )

    logger.info("Export final dataset %s", path_final)
    con.sql(
        f"""
        COPY competencia_03
        TO '{path_final}' (FORMAT PARQUET);
        """
    )


def create_targets(con: duckdb.DuckDBPyConnection) -> None:
    get_argentina_inflation()
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
                    WHEN rank_foto_mes_2 = 0 THEN 'BAJA+1'
                    WHEN rank_foto_mes_2 =-1 THEN 'BAJA+2'
                    ELSE 'CONTINUA'
                END AS clase_ternaria
            FROM competencia_03
        );
        """
    )

    logger.info("Drop ranks")
    con.sql(
        """
        ALTER TABLE competencia_03 DROP COLUMN rank_foto_mes;
        ALTER TABLE competencia_03 DROP COLUMN rank_foto_mes_2;
        """
    )

    logger.info("Creating binary class")
    con.sql(
        """
        CREATE OR REPLACE TABLE competencia_03 AS (
            SELECT
                *,
                CASE
                    WHEN clase_ternaria = 'BAJA+2' THEN 1
                    WHEN clase_ternaria = 'BAJA+1' THEN 1
                    WHEN clase_ternaria = 'CONTINUA' THEN 0
                    ELSE 0
                END AS clase_binaria
            FROM competencia_03
        )
        """
    )


def adjust_inflation(con: duckdb.DuckDBPyConnection) -> None:
    logger.info("Adjusting inflation")
    con.sql(
        f"""
        CREATE OR REPLACE TABLE arg_inflation AS (
            SELECT
                *
            FROM
                read_parquet('{PATH_INFLATION_FINAL}')
        )
        """
    )

    con.sql(
        """
        CREATE OR REPLACE TABLE competencia_03 AS (
            SELECT
                c.*,
                i.indice_inflacion_acumulada
            FROM competencia_03 AS c
            LEFT JOIN arg_inflation AS i
            ON c.foto_mes = i.foto_mes
        );
        """
    )

    with open(INFLATION) as f:
        query = f.read()
        con.sql(f"{query}")

    con.sql(
        """
        ALTER TABLE competencia_03 DROP COLUMN indice_inflacion_acumulada;
        DROP TABLE arg_inflation;
        """
    )
