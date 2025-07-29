executable = generate_ats.sh
getenv = true
environment = "CLUSTER_ID=$(Cluster) PROCESS_ID=$(Process)"
arguments = --checkpoint ./led_2048/checkpoint-19322/ --output_dir 2048_max_input
transfer_executable = false
output = generate_ats_2048.out
error = generate_ats_2048.err
log = generate_ats_2048.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 1600
queue

executable = generate_ats.sh
getenv = true
environment = "CLUSTER_ID=$(Cluster) PROCESS_ID=$(Process)"
arguments = --checkpoint ./led_1024/checkpoint-19322/ --max_input_length 1024 --output_dir 1024_max_input
transfer_executable = false
output = generate_ats_1024.out
error = generate_ats_1024.err
log = generate_ats_1024.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 1600
queue