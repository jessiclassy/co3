executable = minwiki.sh
getenv = true
arguments = --checkpoint ./led_1024/checkpoint-19322/
transfer_executable = false
output = evaluate_1024.$(Cluster).$(Process).out
error = evaluate_1024.$(Cluster).$(Process).err
log = evaluate_1024.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 1500
queue