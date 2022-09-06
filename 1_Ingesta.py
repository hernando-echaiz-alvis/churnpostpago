# Import librerias
import argparse
import time
import pandas as pd
import json
import oracle_lib as ol
import params_lib as pl
import s3_lib as s3l
import os
from datetime import datetime
from os import mkdir


now = datetime.now()

def leer_configuracion():
    ruta_parametros = "./config/ParametrosIngesta.json"
    with open(ruta_parametros) as f:
        try:
            parametrosIngesta = json.load(f)
        except IOError as e:
            e = ("El archivo de configuración no puede ser leido.")
            raise Exception(e)
    return parametrosIngesta


def crea_patron_nombre_tabla(tablePrefix, tableSuffix, periodoEjecutado):
    # anio = periodoEjecutado.substring(0, 4)d
    anio = periodoEjecutado[:4]
    print(anio)
    # mesFinal = periodoEjecutado.substring(4, 2)c
    mesFinal = periodoEjecutado[4:6]
    print(mesFinal)
    # periodo = anio.substring(2, 2)i
    periodoInicial = anio[2:4]
    periodoFinal = anio[2:4]
    mesInicial = ("0" + str(int(mesFinal) - 2))[-2:]

    print("mes inicial "+mesInicial)
    if (mesInicial == '00'):
        periodoInicial = int(periodoInicial) - 1
        print(f"periodo inicial es {periodoInicial} ")
        mesInicial = 12
        print(f"mesInicial es {mesInicial} ")
    elif mesInicial == '-1':
        periodoInicial = int(periodoInicial) - 1
        print(f"periodo inicial es {periodoInicial} ")
        mesInicial = 11
        print(f"mesInicial es {mesInicial} ")
    else:
        print(f"{mesInicial} Mes Inicial Normal")

    tableRange = f"{periodoInicial}{mesInicial}_{periodoFinal}{mesFinal}"
    print(tableRange)
    tableNamePattern = tablePrefix + tableRange + tableSuffix
    print(tableNamePattern)
    return tableNamePattern


# Prepara nombre del dataset
def prepara_nombre(parametrosIngesta, nombreArchivoParametro, periodoEjecutado, rutaArchivoParametro, extensionParametro):
    nombreArchivoParametro = pl.validar_parametros(nombreArchivoParametro, "El parametro nombreArchivo es obligatorio.")
    nombreArchivo = pl.validar_parametros(parametrosIngesta["name"][nombreArchivoParametro], "nombreArchivo no esta definido en el archivo de configuración.")
    periodoEjecutado = pl.validar_parametros(periodoEjecutado, "El periodoEjecutado es obligatorio.")
    rutaArchivoParametro = pl.validar_parametros(rutaArchivoParametro, "El parametro rutaArchivoParametro es obligatorio.")
    rutaArchivo= pl.validar_parametros(parametrosIngesta["paths"][rutaArchivoParametro], "rutaArchivo no esta definido en el archivo de configuración.")
    extensionParametro = pl.validar_parametros(extensionParametro, "El parametro extension es obligatorio.")
    extension = pl.validar_parametros(parametrosIngesta["paths"][extensionParametro], "extension no esta definido en el archivo de configuración.")
    #nombreArchivoPreparado = parametrosIngesta["name"][nombreArchivo] + periodoEjecutado + parametrosIngesta["paths"][extension]
    #nombreArchivoPreparado = nombreArchivo + periodoEjecutado + parametrosIngesta["paths"][extension]
    nombreArchivoPreparado = nombreArchivo + periodoEjecutado + extension
    print(nombreArchivoPreparado)
    rutaArchivoPreparada = os.path.join(rutaArchivo, periodoEjecutado)
    print(rutaArchivoPreparada)
    #print(os.path.join(rutaArchivoPreparada, nombreArchivoPreparado))
    #return os.path.join(rutaArchivoPreparada, nombreArchivoPreparado)
    return rutaArchivoPreparada + '/' + nombreArchivoPreparado

# Cargar data
def prepara_nombre_dataset(periodoEjecutado, parametrosIngesta):
    print("Seteando Parametros Churn_Pospago")
    periodoEjecutado = pl.validar_parametros(periodoEjecutado, "El parametro periodoEjecutado es obligatorio.")
    return prepara_nombre(parametrosIngesta, "Name_Raw_Data", periodoEjecutado, "Ruta_Raw_Data",  "ExtCSV")

# Leer tabla oracle
def leer_tablon_oracle(parametrosIngesta, periodoEjecutado):
    print("Leyendo tabla de oracle")
    userName = pl.validar_parametros(parametrosIngesta["tablon"]["userName"], "El parametro userName es obligatorio.")
    password = pl.validar_parametros(parametrosIngesta["tablon"]["password"], "El parametro password es obligatorio.")
    host = pl.validar_parametros(parametrosIngesta["tablon"]["host"], "El parametro host es obligatorio.")
    port = pl.validar_parametros(parametrosIngesta["tablon"]["port"], "El parametro port es obligatorio.")
    sid = pl.validar_parametros(parametrosIngesta["tablon"]["sid"], "El parametro sid es obligatorio.")
    tablePrefix = pl.validar_parametros(parametrosIngesta["tablon"]["tablePrefix"], "El parametro tableName es obligatorio.")
    tableSuffix = parametrosIngesta["tablon"]["tableSuffix"] #validar_parametros(parametrosIngesta["tablon"]["tableSuffix"], "El parametro tableSuffix es obligatorio.")
    # create table pattern
    tableNamePattern = crea_patron_nombre_tabla(tablePrefix, tableSuffix, periodoEjecutado)
    # create read table code
    rows,col_name = ol.readOracleDatabaseTable(userName, password, host, port, sid, tableNamePattern)
    #rows = pl.validar_parametros(rows, "La tabla no contiene registros.")
    return rows,col_name

def guardar_dataset(nombreDataset, rows,col_name,parametrosIngesta):
    #print(f"Guardando dataset {nombreDataset} con {rows.count()} registros")
    prm_aws_s3_bucket=pl.validar_parametros(parametrosIngesta["s3access"]["aws_s3_bucket"], "El parametro bucket es obligatorio.")
    prm_aws_access_key_id=pl.validar_parametros(parametrosIngesta["s3access"]["aws_access_key_id"], "El parametro access_key_id es obligatorio.")
    prm_aws_secret_access_key=pl.validar_parametros(parametrosIngesta["s3access"]["aws_secret_access_key"], "El parametro secret_access_key es obligatorio.")
    df = pd.DataFrame(rows)
    df.columns = col_name
    print(f"Guardando dataset {nombreDataset} con {df.count()} registros")
    #df.to_csv(nombreDataset, index=False)
    s3l.readAndWriteS3(prm_aws_s3_bucket,prm_aws_access_key_id,prm_aws_secret_access_key,nombreDataset,"w",df)
    print(f"Dataset {nombreDataset} guardado")
    return df

def main():
    parser = argparse.ArgumentParser("ingesta")
    parser.add_argument(
        "--periodo",
        type=str,
        help="Periodo de ingesta"
    )
    args = parser.parse_args()
    periodo = args.periodo
    #print(type(periodo))
    parametrosIngesta = leer_configuracion()
    nombreDataset = prepara_nombre_dataset(periodo, parametrosIngesta)
    rows,col_name = leer_tablon_oracle(parametrosIngesta, periodo)
    guardar_dataset(nombreDataset, rows,col_name,parametrosIngesta)

main()
