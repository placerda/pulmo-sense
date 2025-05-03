import os
import concurrent.futures
from azure.storage.blob import BlobServiceClient
from utils.log_config import get_custom_logger
from collections import defaultdict

my_logger = get_custom_logger('download')

def download_blob(blob_client, download_file_path):
    # my_logger.info(f"[download_blob] Starting downloading blobs to {download_file_path}.")  # Log the start of the download process    
    if not os.path.exists(download_file_path):
        os.makedirs(os.path.dirname(download_file_path), exist_ok=True)
    with open(download_file_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
        my_logger.info(f"[download_blob] Downloaded {download_file_path}.") 

def download_from_blob(storage_account, access_key, container_name, download_path):
    my_logger.info(f"[download_from_blob] Starting downloading blobs to {download_path}.")  # Log the start of the download process
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
            my_logger.info(f"Downloading {download_file_path}.")
            futures.append(executor.submit(download_blob, blob_client, download_file_path))

        for future in concurrent.futures.as_completed(futures):
            future.result()  # Wait for each download to complete, handle exceptions if necessary
        
def download_from_blob_with_access_key(blob_url: str, access_key: str, download_path: str):
    my_logger.info(f"[download_from_blob_with_access_key] Starting downloading blobs to {download_path}.")
    from azure.storage.blob import BlobClient
    blob_client = BlobClient.from_blob_url(blob_url=blob_url, credential=access_key)
    if not os.path.exists(download_path):
        os.makedirs(os.path.dirname(download_path), exist_ok=True)    
    with open(download_path, 'wb') as file:
        data = blob_client.download_blob()
        file.write(data.readall())
    my_logger.info(f"[download_blob] Downloaded {download_path}.") 

def get_patient_directories(container_client, disease_path):
    """
    Retrieves the patient directories for a given disease path.
    """
    patient_dirs = defaultdict(list)
    for blob in container_client.list_blobs(name_starts_with=disease_path):
        parts = blob.name.split('/')
        if len(parts) > 2:  # To ensure we are in a scan folder
            patient_id = parts[1]  # Assumes patient ID is the second part in the path
            patient_dirs[patient_id].append(blob.name)
    return patient_dirs
