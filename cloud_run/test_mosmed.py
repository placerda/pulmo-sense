# %% [markdown]
# ### Test Mosmed Model on Azure ML
# This notebook submits an Azure ML job to test your Mosmed model on the cloud.
# The job will run the modified test_mosmed.py which downloads the dataset from blob storage if needed.

# %%
from dotenv import load_dotenv
from datetime import datetime
import os

# Azure ML imports
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential

# Load environment variables from .env file
load_dotenv()

# Configure the Azure workspace credentials
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group_name = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
    workspace_name=workspace_name,
)

# Create or get the GPU cluster
gpu_compute_target = "gpuclusterinferenceindia"  # Update as needed
try:
    gpu_cluster = ml_client.compute.get(gpu_compute_target)
    print(f"Reusing existing cluster: {gpu_compute_target}")
except Exception:
    print("Creating a new GPU compute target...")
    from azure.ai.ml.entities import AmlCompute
    gpu_cluster = AmlCompute(
        name=gpu_compute_target,
        type="amlcompute",
        size="Standard_ND96amsr_A100_v4",  # Adjust size as needed
        min_instances=1,
        max_instances=4,
        idle_time_before_scale_down=180,
        tier="Dedicated",
    )
    gpu_cluster = ml_client.begin_create_or_update(gpu_cluster).result()
print(f"AMLCompute with name {gpu_cluster.name} is ready (size: {gpu_cluster.size}).")

# Azure ML environment and job setup
custom_env_name = "custom-acpt-pytorch-113-cuda117:12"
env_vars = {
    'AZURE_STORAGE_ACCOUNT': os.getenv("AZURE_STORAGE_ACCOUNT"),
    'AZURE_STORAGE_KEY': os.getenv("AZURE_STORAGE_KEY"),
    'BLOB_CONTAINER': os.getenv("BLOB_CONTAINER"),
}

def get_display_name(base_name):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{base_name} {current_time}"

# ViT
# experiment_name = "test_mosmed_model_vit"
# inputs = {
#     "mosmed_dataset": "mosmed_png_normal",  # Adjust as needed (could be a blob container path)
#     "batch_size": 16,
#     "model_type": "vit",  # Choose from "vgg", "vit", or "lstm_attn"    
#     "model_path": "models/vit_binary_best.pth",
#     "vgg_model_path": "models/vgg_binary_best.pth",
#     "model_uri": "https://myexperiments0584390316.blob.core.windows.net/azureml/ExperimentRun/dcid.khaki_nut_hmj8v1471s/outputs/vit_binary_4epoch_0.00050lr_0.954rec.pth",
#     "vgg_model_uri": "none"
# }

# VGG
experiment_name = "test_mosmed_model_vgg"
inputs = {
    "mosmed_dataset": "mosmed_png",  # Adjust as needed (could be a blob container path)
    "batch_size": 16,
    "model_type": "vgg",  # Choose from "vgg", "vit", or "lstm_attn"    
    "model_path": "models/vgg_binary_best.pth",
    "vgg_model_path": "models/vgg_binary_best.pth",
    "model_uri": "https://myexperiments0584390316.blob.core.windows.net/azureml/ExperimentRun/dcid.khaki_cushion_scg7j7tk4m/outputs/vgg_binary_5epoch_0.00050lr_0.997rec.pth",
    "vgg_model_uri": "none"
}

# LSTM-ATTN
# experiment_name = "test_mosmed_model_vgg"
# inputs = {
#     "mosmed_dataset": "mosmed_png",  # Adjust as needed (could be a blob container path)
#     "batch_size": 16,
#     "model_type": "lstm_attn",  # Choose from "vgg", "vit", or "lstm_attn"    
#     "model_path": "models/vgg_multiclass_best.pth",
#     "vgg_model_path": "models/vgg_multiclass_best.pth",
#     "model_uri": "https://myexperiments0584390316.blob.core.windows.net/azureml/ExperimentRun/dcid.khaki_nut_hmj8v1471s/outputs/vit_binary_4epoch_0.00050lr_0.954rec.pth",
#     "vgg_model_uri": "https://myexperiments0584390316.blob.core.windows.net/azureml/ExperimentRun/dcid.khaki_cushion_scg7j7tk4m/outputs/vgg_binary_5epoch_0.00050lr_0.997rec.pth"
# }

# Construct the command to run the test.
command_str = (
    "python -m scripts.test.test_mosmed "
    "--mosmed_dataset ${{inputs.mosmed_dataset}} "
    "--batch_size ${{inputs.batch_size}} "
    "--model_type ${{inputs.model_type}} "
    "--model_path ${{inputs.model_path}} "
    "--vgg_model_path ${{inputs.vgg_model_path}} "
    "--model_uri ${{inputs.model_uri}} "
    "--vgg_model_uri ${{inputs.vgg_model_uri}} "
)

# Create the Azure ML job command
job = command(
    code="../",  # Adjust this path to the location of your source code relative to this script
    command=command_str,
    compute=gpu_compute_target,
    environment=custom_env_name,
    environment_variables=env_vars,
    inputs=inputs,
    experiment_name=experiment_name,
    display_name=get_display_name(experiment_name),
    tags={key: str(value) for key, value in inputs.items()}
)

submitted_job = ml_client.jobs.create_or_update(job)
job_name = submitted_job.id.split("/")[-1]
print(f"Submitted job {get_display_name(experiment_name)} ({job_name}) to Azure ML")

# %%
