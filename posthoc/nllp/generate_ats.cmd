executable = generate_ats.sh
getenv = true
arguments = --checkpoint ./led_old_impl/checkpoint-19322/ --output_dir ats_2048_max_input
transfer_executable = false
output = generate_ats_validated.$(Cluster).$(Process).out
error = generate_ats_validated.$(Cluster).$(Process).err
log = generate_ats_validated.$(Cluster).$(Process).log
request_memory = 1600
queue