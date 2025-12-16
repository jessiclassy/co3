import argparse
from glob import glob
import pandas as pd
import re
from scipy import stats
import os

METRIC_RE = re.compile(r"bertscore|rouge|fkgl|fkre|fogi|t_")

def filter_output(fname, truncated_doc_ids, input_size):
    # directory = os.path.dirname(fname)
    # longname = directory + "/LONG." + os.path.basename(fname)
    # shortname = directory + "/SHORT." + os.path.basename(fname)

    data = pd.read_csv(fname)
    
    # Iterate over columns for relevant metric columns
    metric_columns = [
        c for c in data.columns if METRIC_RE.search(c)
    ]

    # Partition data
    long_data = data.loc[data.doc_id.isin(truncated_doc_ids)].copy()
    short_data = data.loc[~data.doc_id.isin(truncated_doc_ids)].copy()
    
    sig_metrics = []
    all_metrics = []
    # Log results for each metric
    for metric in metric_columns:
        # print(f"######## {metric} #########")
        t_statistic, p_value = stats.ttest_ind(
            long_data[metric],
            short_data[metric]
        )
        all_metrics.append({
                "metric_name": metric,
                "t_statistic": t_statistic,
                "p_value": p_value,
                "total_avg": data[metric].mean(),
                "long_avg": long_data[metric].mean(),
                "short_avg": short_data[metric].mean(),
                "sig_diff": p_value < 0.05
            })
        
        if p_value < 0.05:
            sig_metrics.append(metric)

        # print(f"T-statistic: {t_statistic}")
        # print(f"P-value: {p_value}")

        # print(f"Total average: {data[metric].mean():.2f}")
        # print(f"Long document average: {long_data[metric].mean():.2f}")
        # print(f"Short document average: {short_data[metric].mean():.2f}")
        # print()
    # long_data.to_csv(longname)
    # short_data.to_csv(shortname)
    # print("Wrote filtered data to CSVs")
    print(f"Statistically significant difference for long/short docs:")
    print(f"{len(sig_metrics)}/{len(metric_columns)}")
    
    if len(sig_metrics):
        metric_df = pd.DataFrame(all_metrics)
        metric_df.to_csv(f"truncation/{input_size}_{os.path.basename(fname)}")
    return

def load_truncated_doc_ids(input_size):
    fname = f"data/billsum_test_truncated_doc_ids_{input_size}_limit.txt"
    content = open(fname, mode="r").readlines()[1:] # ignore header
    return [int(i) for i in content]

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_size", type=int, help="Check for truncation with input size X")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    return args

def main():
    args = load_args()

    # Load truncated document IDs for the given input size
    truncated_doc_ids = load_truncated_doc_ids(args.input_size)

    if args.baseline:
        for baseline_file in glob("../output/baselines/*.csv"):
            print(f"Filtering {baseline_file}...")
            filter_output(baseline_file, truncated_doc_ids, args.input_size)
        return
    
    for output_file in glob(f"../output/*.led-base.*{args.input_size}-512*.csv"):
        print(f"Filtering {output_file}...")
        filter_output(output_file, truncated_doc_ids, args.input_size)
    return

if __name__ == "__main__":
    main()