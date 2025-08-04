executable = nllp_finetune_model.sh
getenv = true
arguments = --checkpoint allenai/led-base-16384 --mode train --trainfile preprocess/data/billsum_clean_train_se3-led-2048-512.csv --batch_size 4 --blank_targets keep
transfer_executable = false
output = nllp_finetune_model_wugNATSS-LED.$(Cluster).out
error = nllp_finetune_model_wugNATSS-LED.$(Cluster).err
log = nllp_finetune_model_wugNATSS-LED.$(Cluster).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
notification = error
queue
