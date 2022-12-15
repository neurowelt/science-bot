#!/bin/bash

echo "Downloading datasets..."
mkdir -p ./data
curl -L https://www.dropbox.com/s/65ff0zb0mqi6bdy/encoding_base_comp.npz > ./data/encoding_base_comp.npz
curl -L https://www.dropbox.com/s/m7axuvh1itsc4qe/knowledge_base.pkl > ./data/knowledge_base.pkl
curl -L https://www.dropbox.com/s/33pu6xma8wwqz6c/entity_base.pkl > ./data/entity_base.pkl

