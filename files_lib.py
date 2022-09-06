def find(bucketname,prefixname):
    import boto3
    s3_conn=boto3.client('s3')
    s3result=s3_conn.list_objects_v2(Bucket=bucketname,Prefix=prefixname)
    
    return s3result["Contents"][0]["Key"].split("/")[-1]
