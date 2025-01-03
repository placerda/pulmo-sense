#!/bin/bash

# Directory to download the files
download_dir="data/ccccii"

# Create the directory if it doesn't exist
mkdir -p "$download_dir"

# List of FTP URLs to download
urls=(
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-1.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-2.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-3.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-4.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-5.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-6.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-7.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-8.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-9.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-10.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-11.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-12.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-13.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-14.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-15.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-16.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-17.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-18.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-19.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-20.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-21.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-22.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-23.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-24.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-25.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-26.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-27.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-28.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-29.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-30.zip"
    "ftp://download.cncb.ac.cn/covid-ct/COVID19-31.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-1.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-2.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-3.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-4.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-5.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-6.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-7.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-8.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-9.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-10.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-11.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-12.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-13.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-14.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-15.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-16.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-17.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-18.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-19.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-20.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-21.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-22.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-23.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-24.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-25.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-26.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-27.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-28.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-29.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-30.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-31.zip"
    "ftp://download.cncb.ac.cn/covid-ct/CP-32.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-1.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-2.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-3.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-4.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-5.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-6.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-7.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-8.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-9.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-10.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-11.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-12.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-13.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-14.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-15.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-16.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-17.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-18.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-19.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-20.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-21.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-22.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-23.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-24.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-25.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-26.zip"
    "ftp://download.cncb.ac.cn/covid-ct/Normal-27.zip"
)

# Loop through each URL and download it to the specified directory
for url in "${urls[@]}"
do
    # Extract the filename from the URL
    filename=$(basename "$url")

    # Get the local file path
    local_file="$download_dir/$filename"

    # Check if the file already exists
    if [ -f "$local_file" ]; then
        # Get the remote file size
        remote_size=$(curl -sI "$url" | grep -i Content-Length | awk '{print $2}' | tr -d '\r')

        # Get the local file size
        local_size=$(stat -c%s "$local_file")

        # Compare the file sizes
        if [ "$local_size" -eq "$remote_size" ]; then
            echo "File $filename already exists and is the same size as the remote file, skipping download."
        else
            echo "File $filename exists but is different in size, re-downloading."
            wget -P "$download_dir" "$url"
        fi
    else
        wget -P "$download_dir" "$url"
    fi
done
