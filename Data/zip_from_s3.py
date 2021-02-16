import boto3
import botocore
from sagemaker import get_execution_role
from zipfile import ZipFile

bucket = 'mytestbucket-image2'
data_location = 's3://mytestbucket-image2/Data-20210216T170804Z-001.zip'
filename = "data.zip"
s3 = boto3.client('s3')
s3.download_file(bucket, location, filename)

zip = ZipFile(filename)
zip.extractall()