############
# LED-base, 1024 input (+ATS)
############
executable = finetune_model.sh
getenv = true
environment = "CLUSTER_ID=$(Cluster); PROCESS_ID=$(Process);"
arguments = --checkpoint allenai/led-base-16384 --mode train --trainfile preprocess/data/billsum_clean_train_se3-led-1024-512_simple.csv --batch_size 4 --blank_targets drop
transfer_executable = false
output = phase_1.$(Cluster).$(Process).out
error = phase_1.$(Cluster).$(Process).err
log = phase_1.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
notification = error
queue

############
# LED-base, 2048 input (+ATS)
############
executable = finetune_model.sh
getenv = true
environment = "CLUSTER_ID=$(Cluster); PROCESS_ID=$(Process);"
arguments = --checkpoint allenai/led-base-16384 --mode train --trainfile preprocess/data/billsum_clean_train_se3-led-2048-512_simple.csv --batch_size 4 --blank_targets drop
transfer_executable = false
output = phase_1.$(Cluster).$(Process).out
error = phase_1.$(Cluster).$(Process).err
log = phase_1.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
notification = error
queue

############
# LED-base, 1024 input (-ATS)
############
executable = finetune_model.sh
getenv = true
environment = "CLUSTER_ID=$(Cluster); PROCESS_ID=$(Process);"
arguments = --checkpoint allenai/led-base-16384 --mode train --trainfile preprocess/data/billsum_clean_train_se3-led-1024-512.csv --batch_size 4 --blank_targets drop
transfer_executable = false
output = phase_1.$(Cluster).$(Process).out
error = phase_1.$(Cluster).$(Process).err
log = phase_1.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
notification = error
queue

############
# LED-base, 2048 input (-ATS)
############
executable = finetune_model.sh
getenv = true
environment = "CLUSTER_ID=$(Cluster); PROCESS_ID=$(Process);"
arguments = --checkpoint allenai/led-base-16384 --mode train --trainfile preprocess/data/billsum_clean_train_se3-led-2048-512.csv --batch_size 4 --blank_targets drop
transfer_executable = false
output = phase_1.$(Cluster).$(Process).out
error = phase_1.$(Cluster).$(Process).err
log = phase_1.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
notification = error
queue