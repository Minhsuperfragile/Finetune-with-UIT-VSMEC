#!/bin/bash

# GitHub repository URL
REPO_URL="https://github.com/Minhsuperfragile/Finetune-with-UIT-VSMEC.git"  # Replace with your GitHub repository URL

# Local directory to clone or pull the repository
LOCAL_DIR="group-4-NLP-DS-2025"  # Replace with your desired local directory name

# Check if the directory already exists
if [ -d "$LOCAL_DIR" ]; then
    echo "Directory exists. Pulling the latest changes..."
    cd "$LOCAL_DIR"
    git pull origin main  # Change 'main' to the default branch if different
else
    echo "Directory doesn't exist. Cloning the repository..."
    git clone "$REPO_URL" "$LOCAL_DIR"
fi