import edec.afterprocessing as ap
import sqlite3
import pandas as pd
from dotenv import load_dotenv
import os
from edec.read_data import FS, DataLake

# Sql which data is taken from
link = "df.db"

# Name of SQL table you want to create
table_name = "cell_fail_claim"

# Failed samples
load_dotenv("../.env.dev")  # take environment variables from .env.
fs = FS(os.environ["DATA_ACCOUNT"])
dl = DataLake(
    os.environ["DATA_ACCOUNT"],  # Account
    os.environ["CLAIMS_DATA_PATH"]  # Path
)

df = pd.read_parquet(
    dl,
    filesystem=fs,
)
fail_all = df["tin_an"].to_numpy()
df_temp = df.loc[df["Description"].isin(["BATTERY, EXCH", "CELL MODULE, EXCH"])]
fail = df_temp["tin_an"].to_numpy()

query = 'SELECT rid, tin_an FROM main_table'
df = ap.load_sql(query, db=link)

# Function that creates labels
df["Label"] = df["tin_an"].isin(fail)
df["Label_All"] = df["tin_an"].isin(fail_all)

conn = sqlite3.connect(link)
df.to_sql(table_name, con=conn, if_exists='replace')
conn.close()
