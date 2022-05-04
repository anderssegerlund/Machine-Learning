import edec.afterprocessing as ap
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
from edec.read_data import FS, DataLake

query = 'SELECT DISTINCT tin_an FROM main_table'
df_tin = ap.load_sql(query, db="df.db")
all_tins = df_tin.to_numpy()
all_tins = [x[0] for x in all_tins]

# Failed samples
load_dotenv("../.env.dev")  # Take environment variables from .env.
fs = FS(os.environ["DATA_ACCOUNT"])
dl = DataLake(
    os.environ["DATA_ACCOUNT"],  # Account
    os.environ["CLAIMS_DATA_PATH"]  # Path
)

df = pd.read_parquet(
    dl,
    filesystem=fs,
)

fail = df["tin_an"].to_numpy()

healthy = [x for x in all_tins if x not in fail]

# Split according to train: 0.6, validation: 0.2, test: 0.2
# Stratified with regard to labels
train, rest = train_test_split(healthy, train_size=0.6, random_state=0)
val, test = train_test_split(rest, test_size=0.5, random_state=0)

train = np.concatenate([train, fail[0:int(0.6 * len(fail))]])
val = np.concatenate([val, fail[int(0.6 * len(fail)):int(0.8 * len(fail))]])
test = np.concatenate([test, fail[int(0.8 * len(fail)):]])

# Create dictionary to store lists
dict_dataset_split_claim = {"train": train, "validation": val,
                            "val": val, "test": test}

# Save dictionary as pickle file
with open('dataset_split_dict_claim_all.pickle', 'wb') as handle:
    pickle.dump((fail, dict_dataset_split_claim), handle,
                protocol=pickle.HIGHEST_PROTOCOL)
