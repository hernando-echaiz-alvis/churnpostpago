# Import librerias
import numpy as np
import gc
import time
import pandas as pd
from os import path, mkdir
import os
import json
import argparse
import params_lib as pl
import files_lib as fl
import s3_lib as s3l
from datetime import datetime, date, timedelta

# Parametros
ini = time.time()
now = datetime.now()


def leer_configuracion():
    ruta_parametros = "./config/ParametrosPrepare.json"
    with open(ruta_parametros) as f:
        try:
            parametrosPrepare = json.load(f)
        except IOError as e:
            e = ("El archivo de configuración no puede ser leido.")
            raise Exception(e)
    return parametrosPrepare


def carga_dataset(parametrosPrepare, periodoEjecutado):
    print("Cargando dataset de zona raw")
    rutaArchivo = pl.validar_parametros(
        parametrosPrepare["paths"]["Ruta_Raw_Data"], "La ruta raw del archivo de parametros no puede ser nula")
    extensionArchivo = pl.validar_parametros(
        parametrosPrepare["paths"]["ExtCSV"], "La ruta raw del archivo de parametros no puede ser nula")
    nombreSalida = pl.validar_parametros(
        parametrosPrepare["name"]["Name_Raw_Data"], "La ruta raw del archivo de parametros no puede ser nula")
    rutaArchivoPreparada = path.join(rutaArchivo, periodoEjecutado)
    prm_aws_s3_bucket=pl.validar_parametros(parametrosPrepare["s3access"]["aws_s3_bucket"], "El parametro bucket es obligatorio.")
    prm_aws_access_key_id=pl.validar_parametros(parametrosPrepare["s3access"]["aws_access_key_id"], "El parametro access_key_id es obligatorio.")
    prm_aws_secret_access_key=pl.validar_parametros(parametrosPrepare["s3access"]["aws_secret_access_key"], "El parametro secret_access_key es obligatorio.")

    files = rutaArchivoPreparada + "/" + nombreSalida + periodoEjecutado + extensionArchivo

    #files=fl.find(prm_aws_s3_bucket,rutaArchivoPreparada)
    print(prm_aws_s3_bucket, rutaArchivoPreparada)
    print(files)
    #print(files,rutaArchivoPreparada)
    if not files.endswith(extensionArchivo):
        raise Exception(f"No se encontraron archivos {extensionArchivo} en la ruta: {rutaArchivoPreparada} del bucket {prm_aws_s3_bucket}")

    #df_ad=s3l.readAndWriteS3(prm_aws_s3_bucket,prm_aws_access_key_id,prm_aws_secret_access_key,f"{rutaArchivoPreparada}/{files}")
    df_ad=s3l.readAndWriteS3(prm_aws_s3_bucket,prm_aws_access_key_id,prm_aws_secret_access_key,files)
    return df_ad


def prepare_dataset(df_ad):
    print("Preparando dataset")
    feats_null = ['REC_CONSWC', 'AMPL_CONSWC', 'REC_TIPIF', 'AMPL_TIPIF']
    for m in feats_null:
        df_ad[m].fillna(90, inplace=True)

    ini = time.time()
    df_ad['FLAG_RENOV'] = np.nan
    df_ad.loc[df_ad['MESES_RENOV'].notnull(), 'FLAG_RENOV'] = 1
    df_ad.loc[df_ad['MESES_RENOV'].isnull(), 'FLAG_RENOV'] = 0
    fin = time.time()
    print('El proceso ha tardado:', (fin-ini)/60, 'Minutos')

    df_ad.replace({'TIPO_DOCUMENTO': {'DNI': 0, 'RUC': 2, 'PASAPORTE': 1},
                   'SEGMENTO_ULTP': {'C': 2, 'B': 1, 'A': 0, 'S': 5, 'PREMIUM': 4, 'D': 3}}, inplace=True)

    df_ad.loc[df_ad['TIPO_DOCUMENTO'] ==
              'CARNET EXTRANJERIA', 'TIPO_DOCUMENTO'] = 1
    df_ad['TIPO_DOCUMENTO'].fillna(0, inplace=True)

    gc.collect()

    df_ad['SEGMENTO_ULTP'].fillna(
        df_ad['SEGMENTO_ULTP'].mode()[0], inplace=True)
    df_ad['TIPO_OPERACION_REC'] = np.where(
        df_ad['TIPO_OPERACION'] == 'RENOVACION',
        1,
        np.where(df_ad['TIPO_OPERACION'] == 'RENOVACION ',
                 1,
                 np.where(df_ad['TIPO_OPERACION'] == 'ACTIVACION PORT IN',
                          2,
                          np.where(df_ad['TIPO_OPERACION'] == 'MIGRACION DE PREPAGO A POSTPAGO',
                                   3,
                                   np.where(df_ad['TIPO_OPERACION'] == 'ACTIVACION',
                                            4, np.where(df_ad['TIPO_OPERACION'] == 'ACTIVACION NUEVA', 4, 5))))))

    df_ad['MOTIVO_CONTRATO_ESTADO_REC'] = np.where(
        df_ad['MOTIVO_CONTRATO_ESTADO'] == 'DESBLOQUEO LINEA POR COBRANZA - PARCIAL',
        1,
        np.where(df_ad['MOTIVO_CONTRATO_ESTADO'] == 'ACTIVACION PORT IN',
                 2,
                 np.where(df_ad['MOTIVO_CONTRATO_ESTADO'] == 'PORT-IN',
                          2,
                          np.where(df_ad['MOTIVO_CONTRATO_ESTADO'] == 'MIGRACION DE PREPAGO A POSTPAGO',
                                   3,
                                   np.where(df_ad['MOTIVO_CONTRATO_ESTADO'] == 'DESBLOQUEO LINEA POR PEDIDO DEL CLIENTE - ROBO',
                                            4, 5)))))

    df_ad['PACK_CHIP_REC'] = np.where(
        df_ad['PACK_CHIP'] == 'P', 1, np.where(df_ad['PACK_CHIP'] == 'C', 2, 3))

    df_ad['PORT_OPE_ORIGEN_DES_REC'] = np.where(df_ad['PORT_OPE_ORIGEN_DES'] == 'Movistar', 1,
                                                np.where(df_ad['PORT_OPE_ORIGEN_DES'] == 'Entel', 2,
                                                         np.where(df_ad['PORT_OPE_ORIGEN_DES'] == 'Bitel', 3,
                                                                  np.where(df_ad['PORT_OPE_ORIGEN_DES'] == 'Flash Mobile', 4, 5))))

    df_ad.EDAD = np.where(df_ad.EDAD.isna(), df_ad.EDAD.median(), df_ad.EDAD)
    df_ad.EDAD = np.where((df_ad.EDAD < 18) | (
        df_ad.EDAD > 90), df_ad.EDAD.median(), df_ad.EDAD)
    return df_ad


def graba_dataset(parametrosPrepare, df_ad, periodoEjecutado):
    print("Grabando dataset de zona analytics")
    rutaSalida = pl.validar_parametros(
        parametrosPrepare["paths"]["Ruta_Analytic_Data"], "La ruta analytics del archivo de parametros no puede ser nula")
    rutaSalidaPreparada = path.join(rutaSalida, periodoEjecutado)
    nombreArchivoSalida = pl.validar_parametros(
        parametrosPrepare["name"]["Name_Analytic_Data"], "El nombre del archivo de salida no puede ser nulo")
    extensionArchivo = pl.validar_parametros(
        parametrosPrepare["paths"]["ExtCSV"], "La extension del archivo de salida no puede ser nula")
    nombreArchivoSalidaFinal = nombreArchivoSalida + \
        periodoEjecutado + extensionArchivo
    nombreArchivoSalidaPreparado = path.join(
        rutaSalidaPreparada, nombreArchivoSalidaFinal)
    #os.makedirs(os.path.dirname(nombreArchivoSalidaPreparado), exist_ok = True)
    prm_aws_s3_bucket = pl.validar_parametros(
        parametrosPrepare["s3access"]["aws_s3_bucket"], "El parametro bucket es obligatorio.")
    prm_aws_access_key_id = pl.validar_parametros(
        parametrosPrepare["s3access"]["aws_access_key_id"], "El parametro access_key_id es obligatorio.")
    prm_aws_secret_access_key = pl.validar_parametros(
        parametrosPrepare["s3access"]["aws_secret_access_key"], "El parametro secret_access_key es obligatorio.")
    #df_ad.to_csv(nombreArchivoSalidaPreparado, index=False)
    s3l.readAndWriteS3(prm_aws_s3_bucket, prm_aws_access_key_id,
                       prm_aws_secret_access_key, nombreArchivoSalidaPreparado, "w", df_ad)


def main():
    parser = argparse.ArgumentParser("prepare")
    parser.add_argument(
        "--periodo",
        type=str,
        help="Periodo de preparación de datos"
    )
    args = parser.parse_args()
    periodo = args.periodo
    parametrosPrepare = leer_configuracion()
    df_ad = carga_dataset(parametrosPrepare, periodo)
    df_ad = prepare_dataset(df_ad)
    graba_dataset(parametrosPrepare, df_ad, periodo)


main()
