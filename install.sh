#!/bin/bash

# Checking system version taken from: https://stackoverflow.com/questions/3466166/how-to-check-if-running-in-cygwin-mac-or-linux
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "I'm sorry, conda is not installed!"
    echo "Please refer to: https://github.com/conda-forge/miniforge"
    exit
fi

# Create conda environment
echo "Creating conda environment for ${machine}"
if [ $machine = "Mac" ]
then
    CONDA_SUBDIR=osx-64 conda create -n chatbot python=3.9
    conda activate chatbot
    conda config --env --set subdir osx-64
else
    conda create -n chatbot python=3.9
fi

# Install packages
echo "Installing Python packages..."
conda install -r requirements.txt -n chatbot

# Solves spacy problem with finding model
echo "Downloading spacy model..."
python -m spacy download en_core_web_sm

echo "Downloading datasets..."
mkdir -p ./data
curl -L https://www.dropbox.com/s/65ff0zb0mqi6bdy/encoding_base_comp.npz > ./data/encoding_base_comp.npz
curl -L https://www.dropbox.com/s/m7axuvh1itsc4qe/knowledge_base.pkl > ./data/knowledge_base.pkl
curl -L https://www.dropbox.com/s/33pu6xma8wwqz6c/entity_base.pkl > ./data/entity_base.pkl