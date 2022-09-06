# Import librerias
import time
import pandas as pd
from os import path, mkdir
import os
import numpy as np
import lightgbm as lgb
import gc
import argparse
import params_lib as pl
import files_lib as fl
import s3_lib as s3l
from os import path
from datetime import datetime
import json

# Parametros
ini = time.time()
now = datetime.now()


def leer_configuracion():
    ruta_parametros = "./config/ParametrosSalida.json"
    with open(ruta_parametros) as f:
        try:
            parametrosSalida = json.load(f)
        except IOError as e:
            e = ("El archivo de configuración no puede ser leido.")
            raise Exception(e)
    return parametrosSalida


def carga_dataset(parametrosSalida, periodoEjecutado):
    print("Cargando dataset de zona raw")
    rutaArchivo = pl.validar_parametros(
        parametrosSalida["paths"]["Ruta_Analytic_Data"], "La ruta raw del archivo de parametros no puede ser nula")
    extensionArchivo = pl.validar_parametros(
        parametrosSalida["paths"]["ExtCSV"], "La ruta raw del archivo de parametros no puede ser nula")
    nombreSalida = pl.validar_parametros(
        parametrosSalida["name"]["Name_Analytic_Data"], "La ruta raw del archivo de parametros no puede ser nula")
    rutaArchivoPreparada = path.join(rutaArchivo, periodoEjecutado)
    prm_aws_s3_bucket = pl.validar_parametros(
        parametrosSalida["s3access"]["aws_s3_bucket"], "El parametro bucket es obligatorio.")
    prm_aws_access_key_id = pl.validar_parametros(
        parametrosSalida["s3access"]["aws_access_key_id"], "El parametro access_key_id es obligatorio.")
    prm_aws_secret_access_key = pl.validar_parametros(
        parametrosSalida["s3access"]["aws_secret_access_key"], "El parametro secret_access_key es obligatorio.")
    files = rutaArchivoPreparada + "/" + nombreSalida + \
        periodoEjecutado + extensionArchivo
    print(prm_aws_s3_bucket, rutaArchivoPreparada)
    print(files)
    if not files.endswith(extensionArchivo):
        raise Exception(
            f"No se encontraron archivos {extensionArchivo} en la ruta: {rutaArchivoPreparada} del bucket {prm_aws_s3_bucket}")
    df_ad = s3l.readAndWriteS3(
        prm_aws_s3_bucket, prm_aws_access_key_id, prm_aws_secret_access_key, files)
    return df_ad


def carga_predict(parametrosSalida, periodoEjecutado):
    print("Cargando dataset de zona raw")
    rutaArchivo = pl.validar_parametros(
        parametrosSalida["paths"]["Ruta_PredictTemp_Data"], "La ruta raw del archivo de parametros no puede ser nula")
    extensionArchivo = pl.validar_parametros(
        parametrosSalida["paths"]["ExtCSV"], "La ruta raw del archivo de parametros no puede ser nula")
    nombreSalida = pl.validar_parametros(
        parametrosSalida["name"]["Name_Predict_Data"], "La ruta raw del archivo de parametros no puede ser nula")
    rutaArchivoPreparada = path.join(rutaArchivo, periodoEjecutado)
    prm_aws_s3_bucket = pl.validar_parametros(
        parametrosSalida["s3access"]["aws_s3_bucket"], "El parametro bucket es obligatorio.")
    prm_aws_access_key_id = pl.validar_parametros(
        parametrosSalida["s3access"]["aws_access_key_id"], "El parametro access_key_id es obligatorio.")
    prm_aws_secret_access_key = pl.validar_parametros(
        parametrosSalida["s3access"]["aws_secret_access_key"], "El parametro secret_access_key es obligatorio.")

    files = rutaArchivoPreparada + "/" + nombreSalida + \
        periodoEjecutado + extensionArchivo

    print(prm_aws_s3_bucket, rutaArchivoPreparada)
    print(files)

    if not files.endswith(extensionArchivo):
        raise Exception(
            f"No se encontraron archivos {extensionArchivo} en la ruta: {rutaArchivoPreparada} del bucket {prm_aws_s3_bucket}")

    df_ad = s3l.readAndWriteS3(
        prm_aws_s3_bucket, prm_aws_access_key_id, prm_aws_secret_access_key, files)
    return df_ad


def preparacion_salida(df_ad, predict_s4t, parametrosScoring, periodoEjecutado):

    df_ad['PROBABILIDAD'] = predict_s4t
    df_ad['ANTIGUEDAD_RANGO'] = np.where(
        df_ad['ANTIGUEDAD'] < 2, "Menos a 2 anios", "De 2 anios a mas")

    df_entregable = df_ad[['PERIODO_EXTR', 'MSISDN', 'EDAD',
                           'ANTIGUEDAD', 'ANTIGUEDAD_RANGO', 'CP_TOTAL_CAT', 'PROBABILIDAD']]
    gc.collect()

    df_entregable.sort_values(
        by=['PROBABILIDAD'], ascending=False, inplace=True)
    df_entregable.reset_index(drop=True, inplace=True)
    df_entregable['DECIL'] = pd.qcut(
        df_entregable['PROBABILIDAD'].rank(method='first'), 10, labels=False)
    df_entregable['DECIL'] = (df_entregable['DECIL'] - 10)*-1

    gc.collect()

    tabla1 = df_entregable.groupby('DECIL').agg(
        {'PROBABILIDAD': ['mean', 'min', 'max', 'count']})
    tabla1

    # Tabla de frecuencias absolutas 202205
    tabla_deciles = pd.crosstab(
        df_entregable.DECIL, df_entregable.ANTIGUEDAD_RANGO, margins='True', margins_name='Total')
    tabla_deciles

    df_entregable['ANTIGUEDAD_MESES'] = df_entregable['ANTIGUEDAD']*12
    gc.collect()

    df_entregable.reset_index(inplace=True)
    df_entregable.head()
    df_entregable.rename(columns={'PERIODO_EXTR': 'PERIODO', 'index': 'PRIORIDAD',
                                  'CP_TOTAL_CAT': 'FLAG_CONSULTA_PREVIA'}, inplace=True)
    gc.collect()

    df_entregable['PRIORIDAD'] = df_entregable['PRIORIDAD']+1
    gc.collect()

    df_entregable.EDAD.describe()
    df_entregable.head(5)

    df_entregable['FLAG_5MESES_ANTIGUEDAD'] = np.nan
    df_entregable.loc[df_entregable['ANTIGUEDAD']
                      <= 0.4, 'FLAG_5MESES_ANTIGUEDAD'] = 1
    df_entregable['FLAG_5MESES_ANTIGUEDAD'].fillna(0, inplace=True)
    gc.collect()

    df_entregable['FLAG_5MESES_ANTIGUEDAD'] = df_entregable['FLAG_5MESES_ANTIGUEDAD'].astype(
        int)
    gc.collect()
    return df_entregable


def guardar_salida(df_salida, parametrosSalida, periodoEjecutado):
    print("Guardando salida de zona analytics")
    rutaArchivo = pl.validar_parametros(
        parametrosSalida["paths"]["Ruta_Predict_Data"], "La ruta predict del archivo no puede ser nula")
    nombreArchivo = pl.validar_parametros(
        parametrosSalida["name"]["Name_Predict_Data"], "El nombre del archivo predict no puede ser nulo")
    extensionArchivo = pl.validar_parametros(
        parametrosSalida["paths"]["ExtCSV"], "La extension del archivo no puede ser nula")
    rutaArchivoPreparada = path.join(
        rutaArchivo, periodoEjecutado, nombreArchivo + periodoEjecutado + extensionArchivo)
    #os.makedirs(os.path.dirname(rutaArchivoPreparada), exist_ok = True)
    ini = time.time()
    df_salida = df_salida[['PERIODO', 'MSISDN', 'EDAD', 'ANTIGUEDAD_RANGO', 'ANTIGUEDAD', 'ANTIGUEDAD_MESES',
                           'PROBABILIDAD', 'PRIORIDAD', 'DECIL', 'FLAG_CONSULTA_PREVIA', 'FLAG_5MESES_ANTIGUEDAD']]
    prm_aws_s3_bucket = pl.validar_parametros(
        parametrosSalida["s3access"]["aws_s3_bucket"], "El parametro bucket es obligatorio.")
    prm_aws_access_key_id = pl.validar_parametros(
        parametrosSalida["s3access"]["aws_access_key_id"], "El parametro access_key_id es obligatorio.")
    prm_aws_secret_access_key = pl.validar_parametros(
        parametrosSalida["s3access"]["aws_secret_access_key"], "El parametro secret_access_key es obligatorio.")
    s3l.readAndWriteS3(prm_aws_s3_bucket, prm_aws_access_key_id,
                       prm_aws_secret_access_key, rutaArchivoPreparada, "w", df_salida)
    fin = time.time()


def main():
    parser = argparse.ArgumentParser("prepare")
    parser.add_argument(
        "--periodo",
        type=str,
        help="Periodo de preparación de datos"
    )
    args = parser.parse_args()
    periodo = args.periodo
    parametrosScoring = leer_configuracion()
    df_ad = carga_dataset(parametrosScoring, periodo)
    df_predict = carga_predict(parametrosScoring, periodo)
    df_salida = preparacion_salida(
        df_ad, df_predict, parametrosScoring, periodo)
    guardar_salida(df_salida, parametrosScoring, periodo)


main()
