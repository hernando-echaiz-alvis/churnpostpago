def readOracleDatabaseTable(userName, password, host, port, sid, tableName):
    """
    Reads the table from the Oracle database.
    """
    import cx_Oracle
    import os
    import sys
    import configparser

    #config = configparser.ConfigParser()
    #config.read('config.ini')
    #userName = "SYSTEM" #config['Oracle']['username']
    #password = "Nirvana1" #config['Oracle']['password']
    #host = "192.168.100.207" #config['Oracle']['host']
    #port = "32518" #config['Oracle']['port']
    #sid = "XE" #config['Oracle']['sid']
    #tableName = "SYSTEM.MFPM_TABLONF_2203_2206_S4"

    dsn = cx_Oracle.makedsn(host, port, sid)
    connection = cx_Oracle.connect(userName, password, dsn)
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM " + tableName)
    rows = cursor.fetchall()
    col_name = [row[0] for row in cursor.description]
    cursor.close()
    connection.close()
    return rows,col_name

def insertOracleDatabaseTableScoring(userName, password, host, port, sid, tableName, rutaArchivo):
    import cx_Oracle
    import csv
    dsn = cx_Oracle.makedsn(host, port, sid)
    connection = cx_Oracle.connect(userName, password, dsn)
    cursor = connection.cursor()
    cursor.setinputsizes(None, 25)
    batch_size = 10000
    with open(rutaArchivo, 'r') as csv_file:
        next(csv_file, None)
        csv_reader = csv.reader(csv_file, delimiter=',')
        sql = f"insert into {tableName} (fecha_carga, msisdn, prob_churn, decil) values (:1, :2, :3, :4)"
        data = []
        for line in csv_reader:
            data.append((line[0], line[1], line[2], line[3]))
            if len(data) % batch_size == 0:
                cursor.executemany(sql, data)
                data = []
        if data:
            cursor.executemany(sql, data)
        connection.commit()    
    cursor.close()
    connection.close()
