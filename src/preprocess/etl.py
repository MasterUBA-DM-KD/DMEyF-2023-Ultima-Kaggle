import logging

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


def extract(con: duckdb.DuckDBPyConnection, path_parquet: str, small: bool = False) -> None:
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

    if small:
        generate_small_dataset(con)


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


def create_features(con: duckdb.DuckDBPyConnection) -> None:
    logger.info("Creating features")
    for i in LAG_FILES + TEND_FILES + MOVING_AVG_FILES + DELTA_FILES:
        with open(i) as f:
            query = f.read()
        logger.info("Creating %s", i)
        con.sql(
            f"""
                        CREATE OR REPLACE TABLE competencia_03 AS (
                            {query}
                        );
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
                    WHEN rank_foto_mes_2 = 0 THEN 'BAJA+2'
                    WHEN rank_foto_mes_2 =-1 THEN 'BAJA+1'
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


def create_clase_binaria(con: duckdb.DuckDBPyConnection, path_binaria: str) -> None:
    in_clause_all = ", ".join([str(i) for i in TRAINING_MONTHS + TEST_MONTH])
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

    logger.info("Export binaria %s", path_binaria)
    con.sql(
        f"""
        COPY competencia_03
        TO '{path_binaria}' (FORMAT PARQUET);
        """
    )


def transform(con: duckdb.DuckDBPyConnection, path_binaria: str) -> None:
    adjust_inflation(con)
    create_features(con)
    create_clase_ternaria(con)
    create_clase_binaria(con, path_binaria)


def load(con: duckdb.DuckDBPyConnection, path_database: str) -> None:
    logger.info("Exporting database")
    con.sql(f"EXPORT DATABASE '{path_database}' (FORMAT PARQUET);")
    logger.info("Closing database")
    con.close()


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
