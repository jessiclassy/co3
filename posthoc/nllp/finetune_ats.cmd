executable = finetune_ats.sh
getenv = true
arguments = --output_dir ats_2048_max_input
transfer_executable = false
output = finetune_ats.$(Cluster).out
error = finetune_ats.$(Cluster).err
log = finetune_ats.$(Cluster).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
queue