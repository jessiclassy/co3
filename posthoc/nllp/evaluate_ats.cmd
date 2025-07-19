executable = evaluate_ats.sh
getenv = true
arguments = --checkpoint ./led/checkpoint-19320/
transfer_executable = false
output = evaluate_ats.$(Cluster).$(Process).out
error = evaluate_ats.$(Cluster).$(Process).err
log = evaluate_ats.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
queue

executable = evaluate_ats.sh
getenv = true
arguments = --checkpoint ./led/checkpoint-19322/
transfer_executable = false
output = evaluate_ats.$(Cluster).$(Process).out
error = evaluate_ats.$(Cluster).$(Process).err
log = evaluate_ats.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
queue

executable = evaluate_ats.sh
getenv = true
arguments = --checkpoint ./led_old_impl/checkpoint-19320/
transfer_executable = false
output = evaluate_ats_validated.$(Cluster).$(Process).out
error = evaluate_ats_validated.$(Cluster).$(Process).err
log = evaluate_ats_validated.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
queue

executable = evaluate_ats.sh
getenv = true
arguments = --checkpoint ./led_old_impl/checkpoint-19322/
transfer_executable = false
output = evaluate_ats_validated.$(Cluster).$(Process).out
error = evaluate_ats_validated.$(Cluster).$(Process).err
log = evaluate_ats_validated.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
queue