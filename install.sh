#!/bin/sh

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "I'm sorry, conda is not installed!"
    echo "Please refer to: https://github.com/conda-forge/miniforge"
    exit
fi

# Create ChatBot environment
echo "Creating the environment..."
CONDA_SUBDIR=osx-64 conda create -n chatbot python=3.9
conda activate chatbot
conda config --env --set subdir osx-64

# Install packages
echo "Installing Python packages..."
pip install -r requirements.txt

# Solves spacy problem with finding model
echo "Downloading spacy model..."
python -m spacy download en_core_web_sm

echo "Downloading datasets..."
mkdir -p ./data
curl -L https://www.dropbox.com/s/65ff0zb0mqi6bdy/encoding_base_comp.npz > ./data/encoding_base_comp.npz
curl -L https://www.dropbox.com/s/m7axuvh1itsc4qe/knowledge_base.pkl > ./data/knowledge_base.pkl
curl -L https://www.dropbox.com/s/33pu6xma8wwqz6c/entity_base.pkl > ./data/entity_base.pkl