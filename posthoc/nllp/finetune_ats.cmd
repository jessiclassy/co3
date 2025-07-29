executable = finetune_ats.sh
getenv = true
arguments = --max_input_length 1024 --output_dir led_1024
transfer_executable = false
output = finetune_ats_1024.$(Cluster).out
error = finetune_ats_1024.$(Cluster).err
log = finetune_ats_1024.$(Cluster).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
queue

executable = finetune_ats.sh
getenv = true
arguments = --max_input_length 2048 --output_dir led_2048
transfer_executable = false
output = finetune_ats_2048.$(Cluster).out
error = finetune_ats_2048.$(Cluster).err
log = finetune_ats_2048.$(Cluster).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
queue