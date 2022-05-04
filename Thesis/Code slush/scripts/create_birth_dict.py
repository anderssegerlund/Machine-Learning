"""
========================================================================================
Script which creates a dictionary containing the first timestamp of each tin_an in the
dataset which is accessed from a sql.
Dictionary is saved as a pickle file named birth_dict.pickle.
Dictionary contains tin_an as key and a pandas timestamp object as value containing the
time of the first sample from that tin_an.
========================================================================================
"""

import sqlite3
import pandas as pd
import pickle

# Path to the sql file containing the data
link = "../scripts/data_sql"
conn = sqlite3.connect(link)

query = 'SELECT DISTINCT tin_an FROM main_table;'
uni_tin = pd.read_sql(query, conn)

key = uni_tin.to_numpy()
key = [elem[0] for elem in key]
dict_tin = {K: None for K in key}

query = 'SELECT tin_an, timestamp FROM main_table;'
df = pd.read_sql(query, conn)
np_all = df.to_numpy()

# Save earliest timestamp for each tin_an in a dictionary
for i in range(len(np_all)):
    tin = np_all[i, 0]
    ts = pd.Timestamp(np_all[i, 1])
    if dict_tin[tin] is None:
        dict_tin[tin] = ts
    else:
        dict_tin[tin] = min(dict_tin[tin], ts)

# Save dictionary as pickle file
with open('birth_dict.pickle', 'wb') as handle:
    pickle.dump(dict_tin, handle, protocol=pickle.HIGHEST_PROTOCOL)
