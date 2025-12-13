# Co3: Controlled Content Coverage for Abstractive Legislative Summarization

Repository for Co3 legislative summarization system; it is evaluated on the [BillSum corpus](https://huggingface.co/datasets/FiscalNote/billsum).

# Prerequisites
## Conda first-time setup
1. Clone repository if it does not already exist
```cmd
git clone https://github.com/jessiclassy/co3.git
```
2. Download miniconda [for your OS](https://www.anaconda.com/docs/getting-started/miniconda/main)

3. Create a new environment:
```cmd
conda create -n co3-env
```
4. Activate the environment to start developing! Yay!
```cmd
conda activate co3-env
```
5. Install all the required packages:
```cmd
pip install -r requirements.txt
```
6. Download the optimized spaCy English language model for evaluation 
```cmd
python -m spacy download en_core_web_sm
```
## Semantic Self-Segmentation
To reproduce the semantic self-segmented data from BillSum documents as implemented in previous work, we execute 
metric learning and semantic self-segmentation using a
submodule in the repository:

```cmd
cd preprocess/se3/
git submodule init
git submodule update
condor_submit learning.cmd
```

## Data Cleaning and reformatting
```cmd
python preprocess/clean.py
```
This script does basic regular expression cleaning of extra whitespace and 
redundant headers. It takes approximately 1-2 minutes to run, before Se3 chunking.

```cmd
python preprocess/reformat_se3_data.py
```
This script converts Se3 output (plaintext) as properly escaped CSV files for easier manipulation downstream. Takes 1-2 minutes to run.

# Experiment Protocol
## Phase 1: Methodological Baseline

Condor jobs for Phase 1 model finetuning and evaluation are detailed in the
executables, `Phase1_Finetune.cmd` and `Phase1_Evaluate.cmd` respectively. On an NVIDIA Quadro 8000 RTX, finetuning takes 10-15 hours, while evaluation takes 10-30 hours.

## Phase 2: Finetuning with Control Tokens

Condor jobs for Phase 2 model finetuning and evaluation are detailed in the
executables, `Phase2_Finetune.cmd` and `Phase2_Evaluate.cmd` respectively. On an NVIDIA Quadro 8000 RTX, finetuning takes 20-40 hours, while evaluation takes 10-30 hours.

## Phase 3: Blank-postprocessing

A rare edge case (<1% of test cases) resulting from Phase 2 finetuning with control tokens is that entire documents can result in blank outputs. Various postprocessing strategies to force summary generation for some $k$ chunks of a given document are implemented in this stage. A Condor job for Phase 3 summary generation and evaluation is detailed in the executable `Phase3_Postprocess.cmd`. On an NVIDIA Quadro 8000 RTX, postprocessing takes less than 1 hour.

Evaluation metrics are "patched" onto the new post-processed summaries:

```cmd
python posthoc/patch_metric.py --file <FILENAME> --metric
 <METRICNAME> --batch_size <BATCHSIZE>
 ```
 
NB: the results for different post-processing strategies are not significantly different.
