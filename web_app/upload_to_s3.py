import boto
import os
access_key = os.environ['AWS_ACCESS_KEY_ID']
access_secret_key = os.environ['AWS_SECRET_ACCESS_KEY']


def upload_file_to_s3(file):
    """
    Saves user uploaded wav file to S3 temporarily for spectrogram precessing.
    """
    conn = boto.connect_s3(access_key, access_secret_key)

    bucket = conn.get_bucket('depression-detect')

    file_object = bucket.new_key(file)
    file_object.set_contents_from_filename(file)
