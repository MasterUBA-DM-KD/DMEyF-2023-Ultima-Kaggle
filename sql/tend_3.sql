SELECT
    *
    ,REGR_SLOPE(active_quarter, cliente_antiguedad) OVER ventana_3 AS active_quarter_slope_6
    ,REGR_SLOPE(cliente_vip, cliente_antiguedad) OVER ventana_3 AS cliente_vip_slope_6
    ,REGR_SLOPE(internet, cliente_antiguedad) OVER ventana_3 AS internet_slope_6
    ,REGR_SLOPE(cliente_edad, cliente_antiguedad) OVER ventana_3 AS cliente_edad_slope_6
    ,REGR_SLOPE(cliente_antiguedad, cliente_antiguedad) OVER ventana_3 AS cliente_antiguedad_slope_6
    ,REGR_SLOPE(mrentabilidad, cliente_antiguedad) OVER ventana_3 AS mrentabilidad_slope_6
    ,REGR_SLOPE(mrentabilidad_annual, cliente_antiguedad) OVER ventana_3 AS mrentabilidad_annual_slope_6
    ,REGR_SLOPE(mcomisiones, cliente_antiguedad) OVER ventana_3 AS mcomisiones_slope_6
    ,REGR_SLOPE(mactivos_margen, cliente_antiguedad) OVER ventana_3 AS mactivos_margen_slope_6
    ,REGR_SLOPE(mpasivos_margen, cliente_antiguedad) OVER ventana_3 AS mpasivos_margen_slope_6
    ,REGR_SLOPE(cproductos, cliente_antiguedad) OVER ventana_3 AS cproductos_slope_6
    ,REGR_SLOPE(tcuentas, cliente_antiguedad) OVER ventana_3 AS tcuentas_slope_6
    ,REGR_SLOPE(ccuenta_corriente, cliente_antiguedad) OVER ventana_3 AS ccuenta_corriente_slope_6
    ,REGR_SLOPE(mcuenta_corriente_adicional, cliente_antiguedad) OVER ventana_3 AS mcuenta_corriente_adicional_slope_6
    ,REGR_SLOPE(mcuenta_corriente, cliente_antiguedad) OVER ventana_3 AS mcuenta_corriente_slope_6
    ,REGR_SLOPE(ccaja_ahorro, cliente_antiguedad) OVER ventana_3 AS ccaja_ahorro_slope_6
    ,REGR_SLOPE(mcaja_ahorro, cliente_antiguedad) OVER ventana_3 AS mcaja_ahorro_slope_6
    ,REGR_SLOPE(mcaja_ahorro_adicional, cliente_antiguedad) OVER ventana_3 AS mcaja_ahorro_adicional_slope_6
    ,REGR_SLOPE(mcaja_ahorro_dolares, cliente_antiguedad) OVER ventana_3 AS mcaja_ahorro_dolares_slope_6
    ,REGR_SLOPE(cdescubierto_preacordado, cliente_antiguedad) OVER ventana_3 AS cdescubierto_preacordado_slope_6
    ,REGR_SLOPE(mcuentas_saldo, cliente_antiguedad) OVER ventana_3 AS mcuentas_saldo_slope_6
    ,REGR_SLOPE(ctarjeta_debito, cliente_antiguedad) OVER ventana_3 AS ctarjeta_debito_slope_6
    ,REGR_SLOPE(ctarjeta_debito_transacciones, cliente_antiguedad) OVER ventana_3 AS ctarjeta_debito_transacciones_slope_6
    ,REGR_SLOPE(mautoservicio, cliente_antiguedad) OVER ventana_3 AS mautoservicio_slope_6
    ,REGR_SLOPE(ctarjeta_visa, cliente_antiguedad) OVER ventana_3 AS ctarjeta_visa_slope_6
    ,REGR_SLOPE(ctarjeta_visa_transacciones, cliente_antiguedad) OVER ventana_3 AS ctarjeta_visa_transacciones_slope_6
    ,REGR_SLOPE(mtarjeta_visa_consumo, cliente_antiguedad) OVER ventana_3 AS mtarjeta_visa_consumo_slope_6
    ,REGR_SLOPE(ctarjeta_master, cliente_antiguedad) OVER ventana_3 AS ctarjeta_master_slope_6
    ,REGR_SLOPE(ctarjeta_master_transacciones, cliente_antiguedad) OVER ventana_3 AS ctarjeta_master_transacciones_slope_6
    ,REGR_SLOPE(mtarjeta_master_consumo, cliente_antiguedad) OVER ventana_3 AS mtarjeta_master_consumo_slope_6
    ,REGR_SLOPE(cprestamos_personales, cliente_antiguedad) OVER ventana_3 AS cprestamos_personales_slope_6
    ,REGR_SLOPE(mprestamos_personales, cliente_antiguedad) OVER ventana_3 AS mprestamos_personales_slope_6
    ,REGR_SLOPE(cprestamos_prendarios, cliente_antiguedad) OVER ventana_3 AS cprestamos_prendarios_slope_6
    ,REGR_SLOPE(mprestamos_prendarios, cliente_antiguedad) OVER ventana_3 AS mprestamos_prendarios_slope_6
    ,REGR_SLOPE(cprestamos_hipotecarios, cliente_antiguedad) OVER ventana_3 AS cprestamos_hipotecarios_slope_6
    ,REGR_SLOPE(mprestamos_hipotecarios, cliente_antiguedad) OVER ventana_3 AS mprestamos_hipotecarios_slope_6
    ,REGR_SLOPE(cplazo_fijo, cliente_antiguedad) OVER ventana_3 AS cplazo_fijo_slope_6
    ,REGR_SLOPE(mplazo_fijo_dolares, cliente_antiguedad) OVER ventana_3 AS mplazo_fijo_dolares_slope_6
    ,REGR_SLOPE(mplazo_fijo_pesos, cliente_antiguedad) OVER ventana_3 AS mplazo_fijo_pesos_slope_6
    ,REGR_SLOPE(cinversion1, cliente_antiguedad) OVER ventana_3 AS cinversion1_slope_6
    ,REGR_SLOPE(minversion1_pesos, cliente_antiguedad) OVER ventana_3 AS minversion1_pesos_slope_6
    ,REGR_SLOPE(minversion1_dolares, cliente_antiguedad) OVER ventana_3 AS minversion1_dolares_slope_6
    ,REGR_SLOPE(cinversion2, cliente_antiguedad) OVER ventana_3 AS cinversion2_slope_6
    ,REGR_SLOPE(minversion2, cliente_antiguedad) OVER ventana_3 AS minversion2_slope_6
    ,REGR_SLOPE(cseguro_vida, cliente_antiguedad) OVER ventana_3 AS cseguro_vida_slope_6
    ,REGR_SLOPE(cseguro_auto, cliente_antiguedad) OVER ventana_3 AS cseguro_auto_slope_6
    ,REGR_SLOPE(cseguro_vivienda, cliente_antiguedad) OVER ventana_3 AS cseguro_vivienda_slope_6
    ,REGR_SLOPE(cseguro_accidentes_personales, cliente_antiguedad) OVER ventana_3 AS cseguro_accidentes_personales_slope_6
    ,REGR_SLOPE(ccaja_seguridad, cliente_antiguedad) OVER ventana_3 AS ccaja_seguridad_slope_6
    ,REGR_SLOPE(cpayroll_trx, cliente_antiguedad) OVER ventana_3 AS cpayroll_trx_slope_6
    ,REGR_SLOPE(mpayroll, cliente_antiguedad) OVER ventana_3 AS mpayroll_slope_6
    ,REGR_SLOPE(mpayroll2, cliente_antiguedad) OVER ventana_3 AS mpayroll2_slope_6
    ,REGR_SLOPE(cpayroll2_trx, cliente_antiguedad) OVER ventana_3 AS cpayroll2_trx_slope_6
    ,REGR_SLOPE(ccuenta_debitos_automaticos, cliente_antiguedad) OVER ventana_3 AS ccuenta_debitos_automaticos_slope_6
    ,REGR_SLOPE(mcuenta_debitos_automaticos, cliente_antiguedad) OVER ventana_3 AS mcuenta_debitos_automaticos_slope_6
    ,REGR_SLOPE(ctarjeta_visa_debitos_automaticos, cliente_antiguedad) OVER ventana_3 AS ctarjeta_visa_debitos_automaticos_slope_6
    ,REGR_SLOPE(mttarjeta_visa_debitos_automaticos, cliente_antiguedad) OVER ventana_3 AS mttarjeta_visa_debitos_automaticos_slope_6
    ,REGR_SLOPE(ctarjeta_master_debitos_automaticos, cliente_antiguedad) OVER ventana_3 AS ctarjeta_master_debitos_automaticos_slope_6
    ,REGR_SLOPE(mttarjeta_master_debitos_automaticos, cliente_antiguedad) OVER ventana_3 AS mttarjeta_master_debitos_automaticos_slope_6
    ,REGR_SLOPE(cpagodeservicios, cliente_antiguedad) OVER ventana_3 AS cpagodeservicios_slope_6
    ,REGR_SLOPE(mpagodeservicios, cliente_antiguedad) OVER ventana_3 AS mpagodeservicios_slope_6
    ,REGR_SLOPE(cpagomiscuentas, cliente_antiguedad) OVER ventana_3 AS cpagomiscuentas_slope_6
    ,REGR_SLOPE(mpagomiscuentas, cliente_antiguedad) OVER ventana_3 AS mpagomiscuentas_slope_6
    ,REGR_SLOPE(ccajeros_propios_descuentos, cliente_antiguedad) OVER ventana_3 AS ccajeros_propios_descuentos_slope_6
    ,REGR_SLOPE(mcajeros_propios_descuentos, cliente_antiguedad) OVER ventana_3 AS mcajeros_propios_descuentos_slope_6
    ,REGR_SLOPE(ctarjeta_visa_descuentos, cliente_antiguedad) OVER ventana_3 AS ctarjeta_visa_descuentos_slope_6
    ,REGR_SLOPE(mtarjeta_visa_descuentos, cliente_antiguedad) OVER ventana_3 AS mtarjeta_visa_descuentos_slope_6
    ,REGR_SLOPE(ctarjeta_master_descuentos, cliente_antiguedad) OVER ventana_3 AS ctarjeta_master_descuentos_slope_6
    ,REGR_SLOPE(mtarjeta_master_descuentos, cliente_antiguedad) OVER ventana_3 AS mtarjeta_master_descuentos_slope_6
    ,REGR_SLOPE(ccomisiones_mantenimiento, cliente_antiguedad) OVER ventana_3 AS ccomisiones_mantenimiento_slope_6
    ,REGR_SLOPE(mcomisiones_mantenimiento, cliente_antiguedad) OVER ventana_3 AS mcomisiones_mantenimiento_slope_6
    ,REGR_SLOPE(ccomisiones_otras, cliente_antiguedad) OVER ventana_3 AS ccomisiones_otras_slope_6
    ,REGR_SLOPE(mcomisiones_otras, cliente_antiguedad) OVER ventana_3 AS mcomisiones_otras_slope_6
    ,REGR_SLOPE(cforex, cliente_antiguedad) OVER ventana_3 AS cforex_slope_6
    ,REGR_SLOPE(cforex_buy, cliente_antiguedad) OVER ventana_3 AS cforex_buy_slope_6
    ,REGR_SLOPE(mforex_buy, cliente_antiguedad) OVER ventana_3 AS mforex_buy_slope_6
    ,REGR_SLOPE(cforex_sell, cliente_antiguedad) OVER ventana_3 AS cforex_sell_slope_6
    ,REGR_SLOPE(mforex_sell, cliente_antiguedad) OVER ventana_3 AS mforex_sell_slope_6
    ,REGR_SLOPE(ctransferencias_recibidas, cliente_antiguedad) OVER ventana_3 AS ctransferencias_recibidas_slope_6
    ,REGR_SLOPE(mtransferencias_recibidas, cliente_antiguedad) OVER ventana_3 AS mtransferencias_recibidas_slope_6
    ,REGR_SLOPE(ctransferencias_emitidas, cliente_antiguedad) OVER ventana_3 AS ctransferencias_emitidas_slope_6
    ,REGR_SLOPE(mtransferencias_emitidas, cliente_antiguedad) OVER ventana_3 AS mtransferencias_emitidas_slope_6
    ,REGR_SLOPE(cextraccion_autoservicio, cliente_antiguedad) OVER ventana_3 AS cextraccion_autoservicio_slope_6
    ,REGR_SLOPE(mextraccion_autoservicio, cliente_antiguedad) OVER ventana_3 AS mextraccion_autoservicio_slope_6
    ,REGR_SLOPE(ccheques_depositados, cliente_antiguedad) OVER ventana_3 AS ccheques_depositados_slope_6
    ,REGR_SLOPE(mcheques_depositados, cliente_antiguedad) OVER ventana_3 AS mcheques_depositados_slope_6
    ,REGR_SLOPE(ccheques_emitidos, cliente_antiguedad) OVER ventana_3 AS ccheques_emitidos_slope_6
    ,REGR_SLOPE(mcheques_emitidos, cliente_antiguedad) OVER ventana_3 AS mcheques_emitidos_slope_6
    ,REGR_SLOPE(ccheques_depositados_rechazados, cliente_antiguedad) OVER ventana_3 AS ccheques_depositados_rechazados_slope_6
    ,REGR_SLOPE(mcheques_depositados_rechazados, cliente_antiguedad) OVER ventana_3 AS mcheques_depositados_rechazados_slope_6
    ,REGR_SLOPE(ccheques_emitidos_rechazados, cliente_antiguedad) OVER ventana_3 AS ccheques_emitidos_rechazados_slope_6
    ,REGR_SLOPE(mcheques_emitidos_rechazados, cliente_antiguedad) OVER ventana_3 AS mcheques_emitidos_rechazados_slope_6
    ,REGR_SLOPE(tcallcenter, cliente_antiguedad) OVER ventana_3 AS tcallcenter_slope_6
    ,REGR_SLOPE(ccallcenter_transacciones, cliente_antiguedad) OVER ventana_3 AS ccallcenter_transacciones_slope_6
    ,REGR_SLOPE(thomebanking, cliente_antiguedad) OVER ventana_3 AS thomebanking_slope_6
    ,REGR_SLOPE(chomebanking_transacciones, cliente_antiguedad) OVER ventana_3 AS chomebanking_transacciones_slope_6
    ,REGR_SLOPE(ccajas_transacciones, cliente_antiguedad) OVER ventana_3 AS ccajas_transacciones_slope_6
    ,REGR_SLOPE(ccajas_consultas, cliente_antiguedad) OVER ventana_3 AS ccajas_consultas_slope_6
    ,REGR_SLOPE(ccajas_depositos, cliente_antiguedad) OVER ventana_3 AS ccajas_depositos_slope_6
    ,REGR_SLOPE(ccajas_extracciones, cliente_antiguedad) OVER ventana_3 AS ccajas_extracciones_slope_6
    ,REGR_SLOPE(ccajas_otras, cliente_antiguedad) OVER ventana_3 AS ccajas_otras_slope_6
    ,REGR_SLOPE(catm_trx, cliente_antiguedad) OVER ventana_3 AS catm_trx_slope_6
    ,REGR_SLOPE(matm, cliente_antiguedad) OVER ventana_3 AS matm_slope_6
    ,REGR_SLOPE(catm_trx_other, cliente_antiguedad) OVER ventana_3 AS catm_trx_other_slope_6
    ,REGR_SLOPE(matm_other, cliente_antiguedad) OVER ventana_3 AS matm_other_slope_6
    ,REGR_SLOPE(ctrx_quarter, cliente_antiguedad) OVER ventana_3 AS ctrx_quarter_slope_6
    ,REGR_SLOPE(Master_delinquency, cliente_antiguedad) OVER ventana_3 AS Master_delinquency_slope_6
    ,REGR_SLOPE(Master_status, cliente_antiguedad) OVER ventana_3 AS Master_status_slope_6
    ,REGR_SLOPE(Master_mfinanciacion_limite, cliente_antiguedad) OVER ventana_3 AS Master_mfinanciacion_limite_slope_6
    ,REGR_SLOPE(Master_Fvencimiento, cliente_antiguedad) OVER ventana_3 AS Master_Fvencimiento_slope_6
    ,REGR_SLOPE(Master_Finiciomora, cliente_antiguedad) OVER ventana_3 AS Master_Finiciomora_slope_6
    ,REGR_SLOPE(Master_msaldototal, cliente_antiguedad) OVER ventana_3 AS Master_msaldototal_slope_6
    ,REGR_SLOPE(Master_msaldopesos, cliente_antiguedad) OVER ventana_3 AS Master_msaldopesos_slope_6
    ,REGR_SLOPE(Master_msaldodolares, cliente_antiguedad) OVER ventana_3 AS Master_msaldodolares_slope_6
    ,REGR_SLOPE(Master_mconsumospesos, cliente_antiguedad) OVER ventana_3 AS Master_mconsumospesos_slope_6
    ,REGR_SLOPE(Master_mconsumosdolares, cliente_antiguedad) OVER ventana_3 AS Master_mconsumosdolares_slope_6
    ,REGR_SLOPE(Master_mlimitecompra, cliente_antiguedad) OVER ventana_3 AS Master_mlimitecompra_slope_6
    ,REGR_SLOPE(Master_madelantopesos, cliente_antiguedad) OVER ventana_3 AS Master_madelantopesos_slope_6
    ,REGR_SLOPE(Master_madelantodolares, cliente_antiguedad) OVER ventana_3 AS Master_madelantodolares_slope_6
    ,REGR_SLOPE(Master_fultimo_cierre, cliente_antiguedad) OVER ventana_3 AS Master_fultimo_cierre_slope_6
    ,REGR_SLOPE(Master_mpagado, cliente_antiguedad) OVER ventana_3 AS Master_mpagado_slope_6
    ,REGR_SLOPE(Master_mpagospesos, cliente_antiguedad) OVER ventana_3 AS Master_mpagospesos_slope_6
    ,REGR_SLOPE(Master_mpagosdolares, cliente_antiguedad) OVER ventana_3 AS Master_mpagosdolares_slope_6
    ,REGR_SLOPE(Master_fechaalta, cliente_antiguedad) OVER ventana_3 AS Master_fechaalta_slope_6
    ,REGR_SLOPE(Master_mconsumototal, cliente_antiguedad) OVER ventana_3 AS Master_mconsumototal_slope_6
    ,REGR_SLOPE(Master_cconsumos, cliente_antiguedad) OVER ventana_3 AS Master_cconsumos_slope_6
    ,REGR_SLOPE(Master_cadelantosefectivo, cliente_antiguedad) OVER ventana_3 AS Master_cadelantosefectivo_slope_6
    ,REGR_SLOPE(Master_mpagominimo, cliente_antiguedad) OVER ventana_3 AS Master_mpagominimo_slope_6
    ,REGR_SLOPE(Visa_delinquency, cliente_antiguedad) OVER ventana_3 AS Visa_delinquency_slope_6
    ,REGR_SLOPE(Visa_status, cliente_antiguedad) OVER ventana_3 AS Visa_status_slope_6
    ,REGR_SLOPE(Visa_mfinanciacion_limite, cliente_antiguedad) OVER ventana_3 AS Visa_mfinanciacion_limite_slope_6
    ,REGR_SLOPE(Visa_Fvencimiento, cliente_antiguedad) OVER ventana_3 AS Visa_Fvencimiento_slope_6
    ,REGR_SLOPE(Visa_Finiciomora, cliente_antiguedad) OVER ventana_3 AS Visa_Finiciomora_slope_6
    ,REGR_SLOPE(Visa_msaldototal, cliente_antiguedad) OVER ventana_3 AS Visa_msaldototal_slope_6
    ,REGR_SLOPE(Visa_msaldopesos, cliente_antiguedad) OVER ventana_3 AS Visa_msaldopesos_slope_6
    ,REGR_SLOPE(Visa_msaldodolares, cliente_antiguedad) OVER ventana_3 AS Visa_msaldodolares_slope_6
    ,REGR_SLOPE(Visa_mconsumospesos, cliente_antiguedad) OVER ventana_3 AS Visa_mconsumospesos_slope_6
    ,REGR_SLOPE(Visa_mconsumosdolares, cliente_antiguedad) OVER ventana_3 AS Visa_mconsumosdolares_slope_6
    ,REGR_SLOPE(Visa_mlimitecompra, cliente_antiguedad) OVER ventana_3 AS Visa_mlimitecompra_slope_6
    ,REGR_SLOPE(Visa_madelantopesos, cliente_antiguedad) OVER ventana_3 AS Visa_madelantopesos_slope_6
    ,REGR_SLOPE(Visa_madelantodolares, cliente_antiguedad) OVER ventana_3 AS Visa_madelantodolares_slope_6
    ,REGR_SLOPE(Visa_fultimo_cierre, cliente_antiguedad) OVER ventana_3 AS Visa_fultimo_cierre_slope_6
    ,REGR_SLOPE(Visa_mpagado, cliente_antiguedad) OVER ventana_3 AS Visa_mpagado_slope_6
    ,REGR_SLOPE(Visa_mpagospesos, cliente_antiguedad) OVER ventana_3 AS Visa_mpagospesos_slope_6
    ,REGR_SLOPE(Visa_mpagosdolares, cliente_antiguedad) OVER ventana_3 AS Visa_mpagosdolares_slope_6
    ,REGR_SLOPE(Visa_fechaalta, cliente_antiguedad) OVER ventana_3 AS Visa_fechaalta_slope_6
    ,REGR_SLOPE(Visa_mconsumototal, cliente_antiguedad) OVER ventana_3 AS Visa_mconsumototal_slope_6
    ,REGR_SLOPE(Visa_cconsumos, cliente_antiguedad) OVER ventana_3 AS Visa_cconsumos_slope_6
    ,REGR_SLOPE(Visa_cadelantosefectivo, cliente_antiguedad) OVER ventana_3 AS Visa_cadelantosefectivo_slope_6
    ,REGR_SLOPE(Visa_mpagominimo, cliente_antiguedad) OVER ventana_3 AS Visa_mpagominimo_slope_6
FROM competencia_03
WINDOW ventana_3 AS (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)
