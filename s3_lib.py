def readAndWriteS3(prm_aws_s3_bucket,prm_aws_access_key_id,prm_aws_secret_access_key,
                   referencepath,operationtypeflag="r",df=""):
    """
    Reads/Write from/to S3 Bucket
    """
    import os
    import boto3
    import pandas as pd
    from io import StringIO
    
    s3_client = boto3.client(
    "s3"
    ,aws_access_key_id=prm_aws_access_key_id
    ,aws_secret_access_key=prm_aws_secret_access_key)
    
    if operationtypeflag=='r':
        response = s3_client.get_object(Bucket=prm_aws_s3_bucket, Key=referencepath)
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        
        if status == 200:
            print(f"Successful S3 get_object response. Status - {status}")
            dfforread = pd.read_csv(response.get("Body"))
            return dfforread
        else:
            print(f"Unsuccessful S3 get_object response. Status - {status}")
            
    else:
        with StringIO() as csv_buffer:
            df.to_csv(csv_buffer, index=False)
            response = s3_client.put_object(
                Bucket=prm_aws_s3_bucket, Key=referencepath, Body=csv_buffer.getvalue()
            )
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        
        if status == 200:
            print(f"Successful S3 put_object response. Status - {status}")
        else:
            print(f"Unsuccessful S3 put_object response. Status - {status}")