import logging
from io import BytesIO

import duckdb
import pandas as pd
import requests

from src.constants import PATH_INFLATION_FINAL, PATH_INFLATION_RAW, URL_INFLATION

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_argentina_inflation() -> None:
    logger.info("Downloading inflation data")
    con = duckdb.connect(database=":memory:", read_only=False)
    response = requests.get(URL_INFLATION)
    xls = pd.ExcelFile(BytesIO(response.content))
    inflacion = pd.read_excel(xls, sheet_name="Índices IPC Cobertura Nacional", skiprows=5, usecols="B:CC")

    logger.info("Processing inflation data")
    inflacion = inflacion.transpose()
    inflacion = inflacion[3]
    inflacion = inflacion.rename("indice_inflacion")
    inflacion = inflacion.dropna()

    inflacion.index.name = "foto_mes"
    inflacion.index = inflacion.index.astype(str).str.replace("-", "").str[:-2]

    inflacion = inflacion.reset_index(drop=False)
    inflacion["foto_mes"] = inflacion["foto_mes"].astype(int)

    inflacion.to_parquet(PATH_INFLATION_RAW)

    con.sql(
        f"""
        CREATE OR REPLACE TABLE arg_inflation AS (
            SELECT
                *
            FROM
                read_parquet('{PATH_INFLATION_RAW}')
        )
        """
    )

    con.sql(
        """
        ALTER TABLE arg_inflation
        ADD COLUMN indice_inflacion_acumulada FLOAT;
        """
    )

    con.sql(
        """
        WITH InflacionAcumulada AS (
            SELECT
                foto_mes,
                indice_inflacion,
                (
                    (
                        SELECT
                            indice_inflacion
                        FROM arg_inflation
                        WHERE foto_mes = 202107) / indice_inflacion
                ) AS inflacion_acumulada
            FROM arg_inflation
        )
        UPDATE arg_inflation
        SET indice_inflacion_acumulada = (
            SELECT
                inflacion_acumulada
            FROM InflacionAcumulada
            WHERE arg_inflation.foto_mes = InflacionAcumulada.foto_mes
        );
        """
    )

    con.sql(
        f"""
        COPY arg_inflation
        TO '{PATH_INFLATION_FINAL}' (HEADER TRUE, DELIMITER ',');
        """
    )

    con.close()
