executable = finetune_ats.sh
getenv = true
arguments = --max_input_length 1024 --output_dir led_1024
transfer_executable = false
output = finetune_ats.$(Cluster).out
error = finetune_ats.$(Cluster).err
log = finetune_ats.$(Cluster).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
queue