import time
import spacy
import lftk
from rouge_score import rouge_scorer
from rouge_score import scoring
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from evaluate import load
from summac.model_summac import SummaCZS
from summac.model_summac import SummaCConv
import argparse

# globals
# load a trained pipeline of your choice from spacy
nlp = spacy.load("en_core_web_sm")
# LFTK feature families of interest
READFORMULA = lftk.search_features(family="readformula", return_format = "list_key")
WORDSENT = lftk.search_features(family="wordsent", return_format = "list_key")
WORDDIFF = lftk.search_features(family="worddiff", return_format = "list_key")

def eval_summac(bill_text:str,gen_text:str) -> dict[str, float|int]:
  """Gets the precision, recall, and fmeasure scores for summac

  Arguments:
    sum_text: the gold summary for a bill
    gen_text: the model generated summary for the same bill
  
  Returns:
    a dictionary where key = ? and value = the score
  """
  for i in range(len(gold_doc)):
    #gold document and gen summary summac score
    gen_score = model_conv.score([bill_text], [gen_text])
    return {"summac_score": gen_score[0]} 

def eval_bert(gold_text:str,gen_text:str) -> dict[str, float|int]:
  """Gets the precision, recall, and fmeasure scores for bert

  Arguments:
    gold_text: the gold summary for a bill
    gen_text: the model generated summary for the same bill
  
  Returns:
    a dictionary where key = bert metric type and value = the score
  """
  bertscore = load("bertscore")
  prediction = [gen_text]
  reference = [gold_text]
  results = bertscore.compute(predictions=prediction, references=reference, model_type="distilbert-base-uncased")
  return {evaltype: score[0] for evaltype, score in results.items()}


def eval_rouge(gold_text:str,gen_text:str) -> dict[str,float|int]:
  """Gets the precision, recall, and fmeasure scores for rouge1,rouge2,rougeL, 
  and rougeLsum

  Arguments:
    gold_text: the gold summary for a bill
    gen_text: the model generated summary for the same bill
  
  Returns:
    a dictionary where key = rouge metric and type and value = the score
  """
  scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL","rougeLsum"],use_stemmer=True)
  results = scorer.score(gold_text,gen_text)

  return {
    "rouge1_precision": results['rouge1'].precision,
    "rouge2_precision": results['rouge2'].precision,
    "rougeL_precision": results['rougeL'].precision,
    "rougeLsum_precision": results['rougeLsum'].precision,
    "rouge1_recall": results['rouge1'].recall,
    "rouge2_recall": results['rouge2'].recall,
    "rougeL_recall": results['rougeL'].recall,
    "rougeLsum_recall": results['rougeLsum'].recall,
    "rouge1_fmeasure": results['rouge1'].fmeasure,
    "rouge2_fmeasure": results['rouge2'].fmeasure,
    "rougeL_fmeasure": results['rougeL'].fmeasure,
    "rougeLsum_fmeasure": results['rougeLsum'].fmeasure
  }


def eval_lftk(text:str, lftk_features:list[str] = READFORMULA, suffix:str = "") -> dict[str,float]:
  """Gets the specified lftk features for a given text input.

  Arguments:
    text: The text whose LFTK properties you'd like to know
    lftk_features: A list of the desired lftk features as strings. Can get this
      from lftk.search_features(...)
    suffix: an optional parameter for adding a suffix to the key names of the
      returned dictionary

  Returns:
     a dictionary with key = lftk feature and value = float or int
  """
  LFTK = lftk.Extractor(docs = nlp(text))
  feature_dict = LFTK.extract(lftk_features)
  return {key+suffix:val for key,val in feature_dict.items()}

def eval_all(
  gold_text: str, 
  gen_text: str,
  sum_text: str
  include_rouge:bool = True,
  include_bert:bool = True,
  include_summac:bool = True,
  include_read_formula:bool = True,
  include_word_sent:bool = True,
  include_word_diff:bool = False,
  skip_gold_lftk:bool = True
) -> dict[str,float|int]:
  """Gets evaluation metrics for a pair of texts. Intended to receive a gold and
  generated summary and output rouge scores for the pair and lftk feature scores
  for each text- using a suffix on key name to denote lftk for generated and lftk 
  for gold. If all 'include_*' arguments set to False, will return an empty dict.

  Arguments:
    gold_data: reference/human-written summary of a bill
    gen_data: model-generated summary of the same bill
    include_rouge: whether to include the rouge family of metrics in the final dict 
    include_read_formula: whether to include the lftk 'read_formula' 
      family of metrics
    include_word_sent: whether to include the lftk 'word_sent' family of metrics
    include_word_diff: whether to include the lftk 'word_diff' family of metrics 
    skip_gold_lftk: flag to indicate if we should skip lftk metrics for 'gold_text'

  Returns:
    a dictionary of all the requested metrics for the given pair of 
    gold and generated text
  """
  results = {}

  # Do summac inclusion
  if include_summac:
    results.update(eval_summac(bill_text,gen_text))

  # Do bert inclusion
  if include_bert:
    results.update(eval_bert(gold_text,gen_text))

  # Do rouge inclusion
  if include_rouge:
    results.update(eval_rouge(gold_text,gen_text))

  # Do lftk feature inclusion
  _readformula = READFORMULA if include_read_formula else []
  _wordsent = WORDSENT if include_word_sent else []
  _worddiff = WORDDIFF if include_word_diff else []
  lftk_features = _readformula + _wordsent + _worddiff
  
  # For gold
  if not skip_gold_lftk:
    results.update(eval_lftk(gold_text,lftk_features,suffix=".GOLD"))

  # For gen
  results.update(eval_lftk(gen_text,lftk_features,suffix=".GEN"))

  print("BERT, Rouge, and LFTK evaluated")
  return results

def get_factuality_scores(text: str) -> dict:
    """
    Gets a dictionary of factuality scores for a summary.
    """
    return None


if __name__ == "__main__":
  data = pd.read_csv("../output/deliverable_4/wugwATSS-pegasus/eval_on_simp.csv", usecols=["predicted_summary"])
  gold = pd.read_csv("gold_lftk.csv", usecols=["summary"])
  bills = pd.read_csv("gold_lftk.csv", usecols=["summary"]) #fix this so it points to bill list
  # gold = load_dataset("FiscalNote/billsum")["test"].to_pandas()
  gold_list = gold.summary.tolist()
  gen_list = data.predicted_summary.tolist()
  bill_list = gold.summary.tolist() #fix this so it points to bill list
  scores = []
  for g, p, b in zip(gold_list, gen_list, bill_list):
    scores.append(eval_all(g, p, b))
  final = pd.DataFrame(scores)
  final.to_csv("wugwATSS-pegasus_eval_with_bert_scores.csv", index=False)

