import duckdb

path_crudo = "datasets/raw/competencia_03_crudo.parquet"
path_small = "datasets/raw/competencia_03_small.parquet"

path_clase_ternaria = "datasets/processed/competencia_03_clase_ternaria.parquet"
path_clase_ternaria_csv = "datasets/processed/competencia_03_clase_ternaria.csv"

path_clase_ternaria_csv_small = "datasets/processed/competencia_03_clase_ternaria_small.csv"

lag_files = [
    "sql/lags_1.sql",
    "sql/lags_3.sql",
    "sql/lags_6.sql",
]

delta_files = [
    "sql/delta_lags_1.sql",
    "sql/delta_lags_3.sql",
    "sql/delta_lags_6.sql",
]


def extract_transform_load():
    duckdb.sql(
        f"""
        CREATE OR REPLACE TABLE competencia_03 AS(
            SELECT
                *
            FROM read_parquet('{path_small}')
        );
        """
    )

    duckdb.sql(
        """
            CREATE OR REPLACE TABLE competencia_03 AS (
                SELECT
                    *,
                    RANK() OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes DESC) AS rank_foto_mes,
                FROM competencia_03
            );
            """
    )

    duckdb.sql(
        """
            CREATE OR REPLACE TABLE competencia_03 AS (
                SELECT
                    *,
                    rank_foto_mes*-1 + 1 AS rank_foto_mes_2
                FROM competencia_03
                ORDER BY foto_mes DESC
            );
            """
    )

    duckdb.sql(
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

    for i in lag_files:
        with open(i) as f:
            query = f.read()
        duckdb.sql(
            f"""
            CREATE OR REPLACE TABLE competencia_03 AS (
                {query}
            );
            """
        )

    for i in delta_files:
        with open(i) as f:
            query = f.read()
        duckdb.sql(
            f"""
            CREATE OR REPLACE TABLE competencia_03 AS (
                {query}
            );
            """
        )

    duckdb.sql(
        """
        ALTER TABLE competencia_03 DROP COLUMN rank_foto_mes;
        """
    )

    duckdb.sql(
        """
        ALTER TABLE competencia_03 DROP COLUMN rank_foto_mes_2;
        """
    )

    duckdb.sql(
        f"""
        COPY competencia_03
        TO '{path_clase_ternaria}' (FORMAT PARQUET);
        """
    )

    duckdb.sql(
        f"""
            COPY competencia_03
            TO '{path_clase_ternaria_csv}' (FORMAT CSV, HEADER);
            """
    )

    duckdb.sql(
        """
        CREATE OR REPLACE TABLE competencia_03_test AS (
            SELECT
                numero_de_cliente,
                foto_mes,
                clase_ternaria,
                mcomisiones	,mcomisiones_lag_1,	mcomisiones_lag_3,	mcomisiones_lag_6
            FROM competencia_03
        );
        """
    )

    duckdb.sql(
        f"""
            COPY competencia_03_test
            TO '{path_clase_ternaria_csv_small}' (FORMAT CSV, HEADER);
            """
    )
