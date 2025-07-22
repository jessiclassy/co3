executable = generate_ats.sh
getenv = true
arguments = --checkpoint ./led/checkpoint-19320/
transfer_executable = false
output = generate_ats.$(Cluster).$(Process).out
error = generate_ats.$(Cluster).$(Process).err
log = generate_ats.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 1600
queue

executable = generate_ats.sh
getenv = true
arguments = --checkpoint ./led/checkpoint-19322/
transfer_executable = false
output = generate_ats.$(Cluster).$(Process).out
error = generate_ats.$(Cluster).$(Process).err
log = generate_ats.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 1600
queue

executable = generate_ats.sh
getenv = true
arguments = --checkpoint ./led_old_impl/checkpoint-19320/
transfer_executable = false
output = generate_ats_validated.$(Cluster).$(Process).out
error = generate_ats_validated.$(Cluster).$(Process).err
log = generate_ats_validated.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 1600
queue

executable = generate_ats.sh
getenv = true
arguments = --checkpoint ./led_old_impl/checkpoint-19322/
transfer_executable = false
output = generate_ats_validated.$(Cluster).$(Process).out
error = generate_ats_validated.$(Cluster).$(Process).err
log = generate_ats_validated.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 1600
queue