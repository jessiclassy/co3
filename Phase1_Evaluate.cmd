############
# LED-base, 1024 input (-ATS)
############
executable = evaluate_model.sh
getenv = true
environment = "CLUSTER_ID=$(Cluster); PROCESS_ID=$(Process);"
arguments = --config_id 0 --base_tokenizer allenai/led-base-16384 \
--checkpoint models/led-base/billsum_clean_train_se3-led-1024-512/drop_blank_targets/1024_512_5_epochs/checkpoint-41832/ \
--mode test \
--testfile preprocess/data/billsum_clean_test_se3-led-1024-512.csv \
--batch_size 4 
transfer_executable = false
output = evaluate_phase_1.$(Cluster).$(Process).out
error = evaluate_phase_1.$(Cluster).$(Process).err
log = evaluate_phase_1.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
notification = error
queue

############
# LED-base, 2048 input (+ATS)
############
executable = evaluate_model.sh
getenv = true
environment = "CLUSTER_ID=$(Cluster); PROCESS_ID=$(Process);"
arguments = --config_id 1 --base_tokenizer allenai/led-base-16384 \
--checkpoint models/led-base/billsum_clean_train_se3-led-2048-512_simple/drop_blank_targets/2048_512_5_epochs/checkpoint-22804/ \
--mode test \
--testfile preprocess/data/billsum_clean_test_se3-led-2048-512.csv \
--batch_size 4
transfer_executable = false
output = evaluate_phase_1.$(Cluster).$(Process).out
error = evaluate_phase_1.$(Cluster).$(Process).err
log = evaluate_phase_1.$(Cluster).$(Process).log
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
request_memory = 3000
notification = error
queue
