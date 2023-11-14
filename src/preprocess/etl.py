import logging
from typing import List

import duckdb
import pandas as pd

from src.constants import (
    DELTA_FILES,
    INFLATION_FILE,
    LAG_FILES,
    MOVING_AVG_FILES,
    PATH_INFLATION_FINAL,
    TEND_FILES,
    TEST_MONTH,
    TRAINING_MONTHS,
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


def transform(
    con: duckdb.DuckDBPyConnection,
    inflation: bool = True,
    lag: bool = True,
    delta_lag: bool = False,
    moving_avg: bool = False,
    trend: bool = False,
) -> None:
    logger.info("Transforming")
    create_clase_ternaria(con)
    create_clase_binaria(con)
    if inflation:
        adjust_inflation(con)
    create_features(con, lag, delta_lag, moving_avg, trend)


def load(con: duckdb.DuckDBPyConnection, path_final: str) -> None:
    in_clause_all = ", ".join([str(i) for i in TRAINING_MONTHS + TEST_MONTH])
    logger.info("Filter dataset to training and test months")
    con.sql(
        f"""
                CREATE OR REPLACE TABLE competencia_03 AS (
                    SELECT
                        *
                    FROM competencia_03
                    WHERE foto_mes IN ({in_clause_all})
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


def create_clase_ternaria(con: duckdb.DuckDBPyConnection) -> None:
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


def create_clase_binaria(con: duckdb.DuckDBPyConnection) -> None:
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

    with open(INFLATION_FILE) as f:
        query = f.read()
        con.sql(f"{query}")

    con.sql(
        """
        ALTER TABLE competencia_03 DROP COLUMN indice_inflacion_acumulada;
        DROP TABLE arg_inflation;
        """
    )


def create_features(
    con: duckdb.DuckDBPyConnection,
    lag: bool = True,
    delta_lag: bool = True,
    moving_avg: bool = False,
    trend: bool = False,
) -> None:
    logger.info("Creating features")
    if lag:
        create_feature(con, LAG_FILES)
    if delta_lag:
        create_feature(con, DELTA_FILES)
    if moving_avg:
        create_feature(con, MOVING_AVG_FILES)
    if trend:
        create_feature(con, TEND_FILES)


def create_feature(con: duckdb.DuckDBPyConnection, queries: List[str]) -> None:
    logger.info("Creating feature")
    for i in queries:
        logger.info("Creating %s", i)
        with open(i) as f:
            query = f.read()
            con.sql(
                f"""
                    CREATE OR REPLACE TABLE competencia_03 AS (
                        {query}
                    );
                """
            )


def get_dataframe(con: duckdb.DuckDBPyConnection, query: str) -> pd.DataFrame:
    logger.info("Querying and converting to dataframe")
    return con.sql(query).to_df()
