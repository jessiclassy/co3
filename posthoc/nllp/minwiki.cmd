executable = minwiki.sh
getenv = true
environment = "CLUSTER_ID=$(Cluster) PROCESS_ID=$(Process)"
arguments = --checkpoint ./led_1024/checkpoint-19322/
transfer_executable = false
output = evaluate_1024.out
error = evaluate_1024.err
log = evaluate_1024.log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 1500
queue