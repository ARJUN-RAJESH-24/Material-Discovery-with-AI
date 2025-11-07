#!/bin/bash
echo "Setting up environment..."
conda env create -f environment.yml || conda env update -f environment.yml
source $(conda info --base)/etc/profile.d/conda.sh
conda activate matdl
echo "Environment ready ✅"
