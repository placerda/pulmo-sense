import os
import concurrent.futures
from azure.storage.blob import BlobServiceClient
from utils.log_config import get_custom_logger
from collections import defaultdict

my_logger = get_custom_logger('download')

def download_blob(blob_client, download_file_path):
    if not os.path.exists(download_file_path):
        os.makedirs(os.path.dirname(download_file_path), exist_ok=True)
        with open(download_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())

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

def download_from_blob_balanced(storage_account, access_key, container_name, download_path, max_patients):
    my_logger.info(f"Starting downloading blobs to {download_path}.")  # Log the start of the download process
    blob_service_client = BlobServiceClient(account_url=f"https://{storage_account}.blob.core.windows.net", credential=access_key)
    container_client = blob_service_client.get_container_client(container_name)
    
    diseases = ['CP', 'NCP', 'Normal']
    total_diseases = len(diseases)
    patients_per_disease = max_patients // total_diseases
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        for disease in diseases:
            disease_path = f"{disease}/"
            patient_dirs = get_patient_directories(container_client, disease_path)
            selected_patients = list(patient_dirs.keys())[:patients_per_disease]
            
            for patient_id in selected_patients:
                for blob_name in patient_dirs[patient_id]:
                    blob_client = container_client.get_blob_client(blob_name)
                    download_file_path = os.path.join(download_path, blob_name)
                    futures.append(executor.submit(download_blob, blob_client, download_file_path))

        for future in concurrent.futures.as_completed(futures):
            future.result()  # Wait for each download to complete, handle exceptions if necessary
        
        my_logger.info(f"Finished downloading blobs to {download_path}.")  # Log the completion of the download process
