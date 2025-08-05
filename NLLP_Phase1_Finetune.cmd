############
#
# LED-base, 1024 input (+ATS)
#
############

executable = nllp_finetune_model.sh
getenv = true
arguments = --checkpoint allenai/led-base-16384 --mode train --trainfile preprocess/nllp_data/billsum_clean_train_se3-led-1024-512_simple.csv --batch_size 4 --blank_targets drop
transfer_executable = false
output = nllp_finetune_model.$(Cluster).out
error = nllp_finetune_model.$(Cluster).err
log = nllp_finetune_model.$(Cluster).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
notification = error
queue

@REM LED-base, 2048 input (+ATS)
executable = nllp_finetune_model.sh
getenv = true
arguments = --checkpoint allenai/led-base-16384 --mode train --trainfile preprocess/nllp_data/billsum_clean_train_se3-led-2048-512_simple.csv --batch_size 4 --blank_targets drop
transfer_executable = false
output = nllp_finetune_model.$(Cluster).out
error = nllp_finetune_model.$(Cluster).err
log = nllp_finetune_model.$(Cluster).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
notification = error
queue

@REM LED-base, 1024 input (-ATS)
executable = nllp_finetune_model.sh
getenv = true
arguments = --checkpoint allenai/led-base-16384 --mode train --trainfile preprocess/nllp_data/billsum_clean_train_se3-led-1024-512.csv --batch_size 4 --blank_targets drop
transfer_executable = false
output = nllp_finetune_model.$(Cluster).out
error = nllp_finetune_model.$(Cluster).err
log = nllp_finetune_model.$(Cluster).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
notification = error
queue

@REM LED-base, 2048 input (-ATS)
executable = nllp_finetune_model.sh
getenv = true
arguments = --checkpoint allenai/led-base-16384 --mode train --trainfile preprocess/nllp_data/billsum_clean_train_se3-led-2048-512.csv --batch_size 4 --blank_targets drop
transfer_executable = false
output = nllp_finetune_model.$(Cluster).out
error = nllp_finetune_model.$(Cluster).err
log = nllp_finetune_model.$(Cluster).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
notification = error
queue