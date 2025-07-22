executable = minwiki.sh
getenv = true
arguments = --checkpoint ./led/checkpoint-19320/
transfer_executable = false
output = evaluate_minwiki.$(Cluster).$(Process).out
error = evaluate_minwiki.$(Cluster).$(Process).err
log = evaluate_minwiki.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 1500
queue

executable = minwiki.sh
getenv = true
arguments = --checkpoint ./led/checkpoint-19322/
transfer_executable = false
output = evaluate_minwiki.$(Cluster).$(Process).out
error = evaluate_minwiki.$(Cluster).$(Process).err
log = evaluate_minwiki.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 1500
queue

executable = minwiki.sh
getenv = true
arguments = --checkpoint ./led_old_impl/checkpoint-19320/
transfer_executable = false
output = evaluate_minwiki_validated.$(Cluster).$(Process).out
error = evaluate_minwiki_validated.$(Cluster).$(Process).err
log = evaluate_minwiki_validated.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 1500
queue

executable = minwiki.sh
getenv = true
arguments = --checkpoint ./led_old_impl/checkpoint-19322/
transfer_executable = false
output = evaluate_minwiki_validated.$(Cluster).$(Process).out
error = evaluate_minwiki_validated.$(Cluster).$(Process).err
log = evaluate_minwiki_validated.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 1500
queue