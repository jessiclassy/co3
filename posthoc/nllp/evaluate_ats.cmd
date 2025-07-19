executable = evaluate_ats.sh
getenv = true
transfer_executable = false
output = evaluate_ats.$(Cluster).out
error = evaluate_ats.$(Cluster).err
log = evaluate_ats.$(Cluster).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
queue

executable = evaluate_ats.sh
getenv = true
arguments = --checkpoint ./led_old_impl
transfer_executable = false
output = evaluate_ats_validated.$(Cluster).out
error = evaluate_ats_validated.$(Cluster).err
log = evaluate_ats_validated.$(Cluster).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
queue