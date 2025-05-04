
# %%
from dotenv import load_dotenv
from datetime import datetime
import os

# Load environment variables from .env file
load_dotenv()

# Azure ML specific imports
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import command

# Configure the Azure workspace and authentication
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
# gpu_compute_target = "gpuclusterwesteurope1"
# gpu_compute_target = "gpuclusterswedencentral2"
# gpu_compute_target = "gpuclustercentralindia3"
# gpu_compute_target = "gpuclustercentralindia004"
# gpu_compute_target = "gpuclusterswedencentral5"
gpu_compute_target = "gpuclustercentralindia5"
fold = "5"


experiment_name = "vgg_binary"
dataset_name = "ccccii"
# fold = "full"
# train_dir = f"ccccii_selected_nonsegmented_train"
# val_dir= f"ccccii_selected_nonsegmented_val"
train_dir = f"ccccii_selected_nonsegmented_fold_{fold}_train"
val_dir= f"ccccii_selected_nonsegmented_fold_{fold}_val"


try:
    gpu_cluster = ml_client.compute.get(gpu_compute_target)
    print(f"You already have a cluster named {gpu_compute_target}, we'll reuse it as is.")
except Exception:
    print("Creating a new GPU compute target...")
    from azure.ai.ml.entities import AmlCompute
    gpu_cluster = AmlCompute(
        name=gpu_compute_target,
        type="amlcompute",
        size="Standard_ND96amsr_A100_v4",
        min_instances=0,
        max_instances=4,
        idle_time_before_scale_down=180,
        tier="Dedicated",
    )
    gpu_cluster = ml_client.begin_create_or_update(gpu_cluster).result()
print(f"AMLCompute with name {gpu_cluster.name} is created, the compute size is {gpu_cluster.size}")

# Azure ML environment and job setup
custom_env_name = "custom-acpt-pytorch-113-cuda117:12"

env_vars = {
    'AZURE_STORAGE_ACCOUNT': os.getenv("AZURE_STORAGE_ACCOUNT"),
    'AZURE_STORAGE_KEY': os.getenv("AZURE_STORAGE_KEY"),
    'BLOB_CONTAINER': os.getenv("BLOB_CONTAINER")
}

def get_display_name(base_name):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{base_name} {current_time}"

# Parameters
inputs = {
    "train_dir": train_dir,
    "val_dir": val_dir,
    "num_epochs": 20,
    "batch_size": 16,
    "learning_rate": 0.0005
}

display_name=get_display_name(experiment_name)

job = command(
    inputs=inputs,
    compute=gpu_compute_target,
    environment=custom_env_name,
    code="../",  # location of source code
    command=(
        "python -m scripts.train.train_vgg_binary "
        "--train_dir ${{inputs.train_dir}} "
        "--val_dir ${{inputs.val_dir}} "
        "--num_epochs ${{inputs.num_epochs}} "
        "--batch_size ${{inputs.batch_size}} "
        "--learning_rate ${{inputs.learning_rate}} "
    ),
    environment_variables=env_vars,
    experiment_name=experiment_name,
    display_name=display_name,
    tags= { 'dataset': dataset_name} 
        | {'fold': fold} 
        | {key: str(value) for key, value in inputs.items()}
)

submitted_job = ml_client.jobs.create_or_update(job)

job_name = submitted_job.id.split("/")[-1]
print(f"Submitted job {display_name} ({job_name}) to Azure ML")

# %%
