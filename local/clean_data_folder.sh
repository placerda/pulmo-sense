#!/bin/bash

# Define the base directory
BASE_DIR="data"

# Find and delete files starting with .azDownload
find "$BASE_DIR" -type f -name ".azDownload*" -exec rm -f {} +

echo "Deletion complete."