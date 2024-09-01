import os
import concurrent.futures
from azure.storage.blob import BlobServiceClient
from utils.log_config import get_custom_logger

my_logger = get_custom_logger('download')

def download_blob(blob_client, download_file_path):
    if not os.path.exists(download_file_path):
        # my_logger.info(f"Downloading {blob_client.blob_name} to {download_file_path}")
        os.makedirs(os.path.dirname(download_file_path), exist_ok=True)
        with open(download_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
    # else:
        # my_logger.info(f"{blob_client.blob_name} already exists at {download_file_path}, skipping download.")

def download_from_blob(storage_account, access_key, container_name, download_path):
    my_logger.info(f"Starting downloading blobs to {download_path}.")  # Log the start of the download process
    blob_service_client = BlobServiceClient(account_url=f"https://{storage_account}.blob.core.windows.net", credential=access_key)
    container_client = blob_service_client.get_container_client(container_name)
    blobs_list = container_client.list_blobs(name_starts_with=download_path)

    # if not os.path.exists(download_path):
    #     os.makedirs(download_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        for blob in blobs_list:
            blob_client = container_client.get_blob_client(blob)
            download_file_path = blob.name
            futures.append(executor.submit(download_blob, blob_client, download_file_path))

        for future in concurrent.futures.as_completed(futures):
            future.result()  # Wait for each download to complete, handle exceptions if necessary
        
        my_logger.info(f"Finished downloading blobs to {download_path}.")  # Log the completion of the download process