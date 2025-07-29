#!/bin/sh
# Turn off parallelism warnings
export TOKENIZERS_PARALLELISM=false
# Log cluster ID, process ID
echo "Running on cluster $CLUSTER_ID, process $PROCESS_ID"
# Default params are sufficient for now
/home2/$USER/miniconda3/envs/573-env/bin/python evaluate_ats_minwiki.py $@