############
# LED-base, ?????? input (??????ATS)
############
executable = nllp_finetune_model.sh
getenv = true
environment = "CLUSTER_ID=$(Cluster); PROCESS_ID=$(Process);"
arguments = --checkpoint allenai/led-base-16384 --mode train --trainfile preprocess/nllp_data/??????? --batch_size 4 --blank_targets ?????????
transfer_executable = false
output = phase_1.$(Cluster).$(Process).out
error = phase_1.$(Cluster).$(Process).err
log = phase_1.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
notification = error
queue