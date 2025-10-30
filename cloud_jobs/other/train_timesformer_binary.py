# %% [markdown]
# ### Train TimeSformer Binary Model
# This script submits an Azure ML job to train the TimeSformer model for binary classification (NCP vs Normal).
# It uses your `scripts/train/train_timesformer_binary.py` entrypoint.

# %%
from dotenv import load_dotenv
from datetime import datetime
import os

# Azure ML SDK imports
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential

# Load environment variables from .env
load_dotenv()

# Workspace configuration
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


# Configure run parameters

# Create or get the GPU cluster
# gpu_compute_target = "gpucluteruk"
gpu_compute_target = "gpuclutercentralindia"
# gpu_compute_target = "gpuclustercentralindia2"
# gpu_compute_target = "gpuclustercentralindia3"
experiment_name = "timesformer_binary"
dataset_name = "ccccii"
# fold = "full"
# train_dir = f"ccccii_selected_nonsegmented_train"
# val_dir= f"ccccii_selected_nonsegmented_val"
fold = "1"
train_dir = f"ccccii_selected_nonsegmented_fold_{fold}_train"
val_dir= f"ccccii_selected_nonsegmented_fold_{fold}_val"

try:
    gpu_cluster = ml_client.compute.get(gpu_compute_target)
    print(f"Reusing existing cluster: {gpu_compute_target}")
except Exception:
    print(f"Creating new GPU cluster: {gpu_compute_target}")
    gpu_cluster = AmlCompute(
        name=gpu_compute_target,
        type="amlcompute",
        size="Standard_ND96amsr_A100_v4",
        min_instances=1,
        max_instances=4,
        idle_time_before_scale_down=180,
        tier="Dedicated",
    )
    gpu_cluster = ml_client.begin_create_or_update(gpu_cluster).result()
print(f"Cluster ready: {gpu_cluster.name} ({gpu_cluster.size})")

# Azure ML environment (must have PyTorch + CUDA)
custom_env_name = "custom-acpt-pytorch-113-cuda117:12"

# Pass-through environment variables for data download
env_vars = {
    "AZURE_STORAGE_ACCOUNT": os.getenv("AZURE_STORAGE_ACCOUNT"),
    "AZURE_STORAGE_KEY": os.getenv("AZURE_STORAGE_KEY"),
    "BLOB_CONTAINER": os.getenv("BLOB_CONTAINER"),
}

print(" key: ", os.getenv("AZURE_STORAGE_KEY")[:5], "..." if len(os.getenv("AZURE_STORAGE_KEY")) > 5 else "")

def get_display_name(base_name: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{base_name} {ts}"


# Inputs to the training script
inputs = {
    "train_dir": train_dir,
    "val_dir": val_dir,
    "num_epochs": 20,
    "batch_size": 16,
    "learning_rate": 0.0005,
    "sequence_length": 30,    
}

job = command(
    inputs=inputs,
    compute=gpu_compute_target,
    environment=custom_env_name,
    code="../",  # point to repo root so `scripts/train/train_timesformer_binary.py` is on PYTHONPATH
    command=(
        "python -m scripts.train.train_timesformer_binary "
        "--train_dir ${{inputs.train_dir}} "
        "--val_dir ${{inputs.val_dir}} "
        "--num_epochs ${{inputs.num_epochs}} "
        "--batch_size ${{inputs.batch_size}} "
        "--learning_rate ${{inputs.learning_rate}} "
        "--sequence_length ${{inputs.sequence_length}} "        
    ),      
    environment_variables=env_vars,
    experiment_name=experiment_name,
    display_name=get_display_name(experiment_name),
    tags= { 'dataset': dataset_name} 
        | {'fold': fold} 
        | {key: str(value) for key, value in inputs.items()}
)

submitted_job = ml_client.jobs.create_or_update(job)
job_id = submitted_job.name or submitted_job.id
print(f"Submitted job {get_display_name(experiment_name)} (ID: {job_id})")

# %%
