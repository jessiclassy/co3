#!/bin/sh
# Turn off parallelism warnings
export TOKENIZERS_PARALLELISM=false
# Default params are sufficient for now
/home2/$USER/miniconda3/envs/573-env/bin/python generate_ats_billsum.py $@