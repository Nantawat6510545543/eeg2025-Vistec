import boto3
from botocore import UNSIGNED
from botocore.config import Config
from boto3.s3.transfer import TransferConfig
import os
import sys

class ProgressBar:
    def __init__(self, filename, total):
        self._filename = filename
        self._seen_so_far = 0
        self._total = total
        self._last_percent = -1

    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        percent = int((self._seen_so_far / self._total) * 100)
        if percent != self._last_percent:
            self._last_percent = percent
            sys.stdout.write(
                f"\rDownloading {self._filename}... {percent}% ({self._seen_so_far}/{self._total} bytes)"
            )
            sys.stdout.flush()
        if self._seen_so_far >= self._total:
            print("\nDone.")


s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket = 'fcp-indi'
prefix = 'data/Projects/HBN/BIDS_EEG/cmi_bids_R1/sub-NDARAC904DMU'
local_base_dir = "cmi_bids_R1"

# List all objects under prefix
paginator = s3.get_paginator("list_objects_v2")
pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

for page in pages:
    for obj in page.get("Contents", []):
        key = obj["Key"]
        local_path = os.path.join(local_base_dir, os.path.relpath(key, prefix))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        print(f"Preparing to download: {key} → {local_path}")
        size = obj["Size"]
        progress = ProgressBar(local_path, size)

        s3.download_file(bucket, key, local_path, Callback=progress)
