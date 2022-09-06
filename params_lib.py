def validar_parametros(parametro, mensajeError):
    if not parametro:
        e = (mensajeError)
        raise Exception(e)
    return parametro