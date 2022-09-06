#Import librerias
import time
import pandas as pd
from os import path,mkdir
import os
#import numpy as np
#import lightgbm as lgb
import gc
import joblib
import argparse
import params_lib as pl
import files_lib as fl
from os import path
from datetime import datetime
import json
import s3_lib as s3l

#Parametros
ini = time.time()
now = datetime.now()

def leer_configuracion():
    ruta_parametros = "./config/ParametrosScoring.json"
    with open(ruta_parametros) as f:
        try:
            parametrosScoring = json.load(f)
        except IOError as e:
            e = ("El archivo de configuración no puede ser leido.")
            raise Exception(e)
    return parametrosScoring


def carga_dataset(parametrosScoring, periodoEjecutado):
    print("Cargando dataset de zona raw")
    rutaArchivo = pl.validar_parametros(
        parametrosScoring["paths"]["Ruta_Analytic_Data"], "La ruta raw del archivo de parametros no puede ser nula")
    extensionArchivo = pl.validar_parametros(
        parametrosScoring["paths"]["ExtCSV"], "La ruta raw del archivo de parametros no puede ser nula")
    nombreSalida = pl.validar_parametros(
        parametrosScoring["name"]["Name_Analytic_Data"], "La ruta raw del archivo de parametros no puede ser nula")
    rutaArchivoPreparada = path.join(rutaArchivo, periodoEjecutado)
    prm_aws_s3_bucket = pl.validar_parametros(
        parametrosScoring["s3access"]["aws_s3_bucket"], "El parametro bucket es obligatorio.")
    prm_aws_access_key_id = pl.validar_parametros(
        parametrosScoring["s3access"]["aws_access_key_id"], "El parametro access_key_id es obligatorio.")
    prm_aws_secret_access_key = pl.validar_parametros(
        parametrosScoring["s3access"]["aws_secret_access_key"], "El parametro secret_access_key es obligatorio.")

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

def prepara_nombre_modelo(parametrosScoring):
    nombreModelo = pl.validar_parametros(parametrosScoring["name"]["Name_Analytic_Model"], "El nombre del modelo no puede ser nulo")
    rutaModelo = pl.validar_parametros(parametrosScoring["paths"]["Ruta_Analytic_Model"], "La ruta del modelo no puede ser nula")
    return path.join(rutaModelo, nombreModelo)

def scoring (df_ad, parametrosScoring):
    print("Ejecutando Scoring")
    nombreModelo = prepara_nombre_modelo(parametrosScoring)
    train_columnsf = [
    'CP_TOTAL_PROM',
    'ANTIGUEDAD',
    'REC_TIPIF',
    'CP_TOTAL_CAT',
    'TIPO_OPERACION_REC',
    'PROM_SCORE_PAGO',
    'TIEMPO_CONTRATO_MESES',
    'VF2_244_ULTP',
    'PORT_OPE_ORIGEN_DES_REC',
    'DIAS_INAC_TRAF_DAT_ULTP',
    'CANT_LLAM_SAL_ENT_PROM',
    'MOTIVO_CONTRATO_ESTADO_REC',
    'REC_CONSWC',
    'OTROS',
    'DURAC_LLAM_ENTR_CLA_ULTP',
    'CF_FACT_CTA_ULTP',
    'PACK_CHIP_REC',
    'FLAG_RENOV',
    'DIAS_INAC_TRAF_DAT_PROM',
    'PROM_NAVEGACION_IVR_ATEND',
    'MB_GRATUITOS_ULTP',
    'INFORMACION_RETENCION_PREVENTIVA',
    'EDAD',
    'MB_FACEBOOK_ULTP',
    'INFORMACION_TRANSFERENCIAS',
    'LIMITE_CREDITOS_PROM',
    'SEGMENTO_ULTP']

    #Código que no se usa.
    categorical_featuresf= ["SEGMENTO_ULTP",
                           "TIPO_OPERACION_REC",
                           "MOTIVO_CONTRATO_ESTADO_REC",
                           "PACK_CHIP_REC",
                           "PORT_OPE_ORIGEN_DES_REC"]
    for m in categorical_featuresf:
        df_ad[m] = df_ad[m].astype(float)
        gc.collect()

    categorical_indexf= [train_columnsf.index(x) for x in categorical_featuresf]

    #Ejecución del modelo
    model_s4t = joblib.load(nombreModelo)
    ini = time.time()
    predict_s4t = pd.DataFrame(model_s4t.predict(df_ad[train_columnsf], num_iteration=model_s4t.best_iteration))
    gc.collect()
    fin = time.time()
    print('El proceso ha tardado: ', (fin-ini)/60, 'Minutos')
    print(type(predict_s4t))
    predict_s4t.columns = ["PROBABILIDAD"]

    return predict_s4t


def prepara_nombre_salida(parametrosScoring, periodoEjecutado):
    nombreSalida = pl.validar_parametros(
        parametrosScoring["name"]["Name_PredictTemp_Data"], "El nombre del archivo de salida no puede ser nulo")
    rutaSalida = pl.validar_parametros(
        parametrosScoring["paths"]["Ruta_PredictTemp_Data"], "La ruta del archivo de salida no puede ser nula")
    extensionArchivo = pl.validar_parametros(
        parametrosScoring["paths"]["ExtCSV"], "La extension del archivo de salida no puede ser nula")
    rutaSalidaPreparada = path.join(rutaSalida, periodoEjecutado)
    print(path.join(rutaSalidaPreparada, nombreSalida,
                    periodoEjecutado, extensionArchivo))

    return path.join(rutaSalidaPreparada, nombreSalida + periodoEjecutado + extensionArchivo)

def guardar_salida(predict_s4t, parametrosScoring, periodoEjecutado):
    print("Guardando salida")
    nombreSalida = prepara_nombre_salida(parametrosScoring, periodoEjecutado)
    #os.makedirs(os.path.dirname(nombreSalida), exist_ok = True)
    prm_aws_s3_bucket=pl.validar_parametros(parametrosScoring["s3access"]["aws_s3_bucket"], "El parametro bucket es obligatorio.")
    prm_aws_access_key_id=pl.validar_parametros(parametrosScoring["s3access"]["aws_access_key_id"], "El parametro access_key_id es obligatorio.")
    prm_aws_secret_access_key=pl.validar_parametros(parametrosScoring["s3access"]["aws_secret_access_key"], "El parametro secret_access_key es obligatorio.")
    s3l.readAndWriteS3(prm_aws_s3_bucket,prm_aws_access_key_id,prm_aws_secret_access_key,nombreSalida,"w",predict_s4t)
    #predict_s4t.to_csv(nombreSalida, index=False)

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
    df_predict = scoring(df_ad, parametrosScoring)
    guardar_salida(df_predict, parametrosScoring, periodo)

main()
