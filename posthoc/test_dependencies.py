import torch
from alignscore import AlignScore
from summac.model_summac import SummaCConv
import sys
sys.path.insert(0, "..")
from eval.eval_metrics import eval_all

curr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on ", curr_device)

document = """Scientists are studying Mars to learn about the Red Planet and find landing sites for future missions.
One possible site, known as Arcadia Planitia, is covered instrange sinuous features.
The shapes could be signs that the area is actually made of glaciers, which are large masses of slow-moving ice.
Arcadia Planitia is in Mars' northern lowlands."""

summary1 = "There are strange shape patterns on Arcadia Planitia. The shapes could indicate the area might be made of glaciers. This makes Arcadia Planitia ideal for future missions."
align_scorer = AlignScore(model='roberta-base', batch_size=32, device=curr_device, ckpt_path='../AlignScore-base.ckpt', evaluation_mode='nli_sp')
model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device=curr_device, agg="mean")

def test_factuality(doc, summary):
    # TEST ALIGNSCORE
    align_score = align_scorer.score(contexts=[doc], claims=[summary])

    # TEST SUMMAC
    score_conv1 = model_conv.score([doc], [summary])
    return {
        "alignscore": align_score[0],
        "summac": score_conv1["scores"][0]
    }

metrics = eval_all(document, summary1)
factuality = test_factuality(document, summary1)

metrics.update(factuality)

print(metrics)