import pandas as pd

# INPUT
sourcename = "../preprocess/data/billsum_clean_test_se3-t5-512-512"
simplified = sourcename + "_simple"
outname = simplified + "_patched"

source_df = pd.read_csv(sourcename + ".csv")
simp_df = pd.read_csv(simplified + ".csv")

# patch_idx = simp_df[simp_df.isnull().any(axis=1)].index.tolist()
simp_df.update(source_df, overwrite=False)
simp_df.to_csv(outname + ".csv")