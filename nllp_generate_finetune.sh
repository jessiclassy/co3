#!/bin/bash

# Check if config file was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <finetune_strategy>.config"
    exit 1
fi

CONFIG_FILE="configs/finetune/$1.config"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $1 not found under configs/finetune"
    exit 1
fi

# Parse the config file
source $CONFIG_FILE

# Validate required parameters
if [ -z "$PLATFORM" ]; then
    echo "Error: PLATFORM not specified in config file"
    exit 1
fi

if [ -z "$CHECKPOINT" ]; then
    echo "Error: CHECKPOINT not specified in config file"
    exit 1
fi

if [ -z "$MODE" ]; then
    echo "Error: MODE not specified in config file"
    exit 1
fi

if [ -z "$BLANKTARGETS"]; then
    echo "Error: BLANKTARGETS not specified in config file"
    exit 1
fi

# Log mode parameter
if [ $MODE = "train" ] ; then
    echo "Finetuning on provided dataset"
else
    echo "Running end-to-end system"
fi

# Generate files based on PLATFORM
case "$PLATFORM" in
    patas)
        echo "Preparing to run finetuning on Patas..."
        # Generate Condor .cmd file
        output_file="nllp_finetune_model.$1.cmd"
        
        cat > "$output_file" <<EOF
executable = nllp_finetune_model.sh
getenv = true
arguments = --checkpoint $CHECKPOINT --mode $MODE --trainfile $TRAINFILE --batch_size $BATCH_SIZE --blank_targets $BLANKTARGETS
transfer_executable = false
output = nllp_finetune_model.\$(Cluster).out
error = nllp_finetune_model.\$(Cluster).err
log = nllp_finetune_model.\$(Cluster).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
notification = error
queue
EOF
        
        echo "Generated Condor submit file: $output_file"
        condor_submit $output_file
        ;;
        
        
    hyak)
        # Generate Slurm .sbatch file
        output_file="finetune_model.slurm"
        
        cat > "$output_file" <<EOF
#!/bin/bash
#SBATCH --job-name=finetune_model
#SBATCH --output=out/%x/%j.out
#SBATCH --error=error/%x/%j.err
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a40:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --mail-user=$USER@uw.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

apptainer exec --cleanenv --bind /gscratch /gscratch/scrubbed/jcmw614/project.sif \
/gscratch/scrubbed/jcmw614/envs/573-env/bin/python finetune_model.py \
--checkpoint $CHECKPOINT --mode $MODE --trainfile $TRAINFILE \
--batch_size $BATCH_SIZE 
EOF
        
        echo "Generated Slurm submit file: $output_file"
        sbatch $outputfile
        ;;
        
    *)
        echo "Error: Unknown PLATFORM '$PLATFORM'. Supported PLATFORMs are 'patas' and 'hyak'"
        exit 1
        ;;
esac