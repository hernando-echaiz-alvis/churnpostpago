def validateDatabaseTable():

    import cx_Oracle
    import os
    from datetime import datetime,date
    from dateutil.relativedelta import relativedelta
    userName=os.environ.get("oracleUser")
    password=os.environ.get("oraclePwd")
    host=os.environ.get("oracleHostIp")
    port=os.environ.get("oraclePort")
    sid=os.environ.get("oracleDbName")
    tableSchemaName=os.environ.get("oracleSchema")
    tableNamePattern=os.environ.get("oracleTableName")
    valueOfMonth=os.environ.get("monthofProcessValue")
    stringOfMonth = str(valueOfMonth)
    shortMonth = str(stringOfMonth)[2:]
    prevMonthValue = (datetime.strptime(stringOfMonth + "01","%Y%m%d") + relativedelta(days=-1)).strftime("%Y%m"))
    month_2_Value = (datetime.strptime(prevMonthValue + "01","%Y%m%d") + relativedelta(days=-1)).strftime("%Y%m"))
    month_3_Value = (datetime.strptime(month_2_Value + "01","%Y%m%d") + relativedelta(days=-1)).strftime("%Y%m"))
    tableName = tableNamePattern.format(oracleSchema=tableSchemaName,monthOfProcess_n=month_2_Value[2:],
                                      monthOfProcess=shortMonth)
    prevMonthtableName = tableNamePattern.format(oracleSchema=tableSchemaName,monthOfProcess_n=month_3_Value[2:],
                                        monthOfProcess=prevMonthValue[2:])    
    flagRecords=os.environ.get("recordsControlFlag")
    recordsVariationLimit=os.environ.get("recordsVariationThreshold")
    dsn = cx_Oracle.makedsn(host, port, sid)
    connection = cx_Oracle.connect(userName, password, dsn)
    cursor = connection.cursor()
    cursor.execute(f"SELECT COUNT(1) FROM ALL_TABLES WHERE CONCAT(CONCAT(OWNER,'.'),\
                   TABLE_NAME)='{tableName.upper()}'")
    firstresult=cursor.fetchone()[0]
    cursor.execute(f"SELECT COUNT(1) FROM ALL_TABLES WHERE CONCAT(CONCAT(OWNER,'.'),\
                   TABLE_NAME)='{prevMonthtableName.upper()}'")    
    secondresult=cursor.fecthome()[0]
    if firstresult==0:
        msg=f"La tabla {tableName} no existe por lo que no se podrá continuar \
        con la ejecución mensual"
    else:
        cursor.execute("SELECT COUNT(1) FROM " + tableName)
        fourthresult=cursor.fecthome()[0]
        if flagRecords==1:
            if secondresult==0:
                msg=f"La tabla {prevMonthtableName} no existe por lo que no se continuará \
                con la ejecución mensual ya que no es posible validar la tendencia \
                en el número de registros de la fuente"
            else:
                cursor.execute("SELECT COUNT(1) FROM " + prevMonthtableName)
                thirdresult=cursor.fecthome()[0]
                if thirdresult==0:
                    msg=f"La tabla {prevMonthtableName} no cuenta con registros por lo que no se continuará \
                    con la ejecución mensual ya que no es posible validar la tendencia \
                    en el número de registros de la fuente"
                else:
                    recordsvariationrate=round((fourthresult-thirdresult)*100/thirdresult,2)
                    if recordsvariationrate<=recordsVariationLimit:
                        msg="1"
                    else:
                        msg=f"La variación porcentual en el número de registros es del \
                        {str(recordsvariationrate)}%,el cual excede el umbral \
                        permitido el cual es del {recordsVariationLimit}%"
        else:
            if fourthresult>0:
                msg="1"
            else:
                msg=f"La tabla {tableName} no cuenta con registros por lo que no podrá continuar \
                con la ejecución mensual"
                
    with open("/airflow/xcom/return.json","w") as file:
        file.write(msg)
 
 def main():
     validateDatabaseTable()

main()     