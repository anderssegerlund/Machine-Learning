import edec.afterprocessing as ap
from sklearn.model_selection import train_test_split
import pickle


query = 'SELECT DISTINCT tin_an FROM main_table'
df_tin = ap.load_sql(query, db="df.db")

# Labeling data, results end up in cell_fail table in column 'Fail'
ap.annotate_db(sql_db_name='df.db', threshold_soc=9.0, threshold_volt=0.01)

# Retrieve the tin numbers that have at least 1 failed sample (both)
query = 'SELECT DISTINCT main_table.tin_an FROM main_table \
INNER JOIN cell_fail \
ON main_table.rid = cell_fail.rid \
WHERE cell_fail.Fail == 1'
df_tin_fail = ap.load_sql(query, db="df.db")

# Retrieve the tin numbers that have at least 1 failed sample (soc)
query = 'SELECT DISTINCT main_table.tin_an FROM main_table \
INNER JOIN cell_fail \
ON main_table.rid = cell_fail.rid \
WHERE cell_fail.Fail_Soc == 1 AND cell_fail.Fail_Volt == 0'
df_tin_fail_soc = ap.load_sql(query, db="df.db")

# Retrieve the tin numbers that have at least 1 failed sample (volt)
query = 'SELECT DISTINCT main_table.tin_an FROM main_table \
INNER JOIN cell_fail \
ON main_table.rid = cell_fail.rid \
WHERE cell_fail.Fail_Volt == 1 AND cell_fail.Fail_Soc == 0'
df_tin_fail_volt = ap.load_sql(query, db="df.db")


# Split according to train: 0.5, validation: 0.2, test: 0.15, test_final: 0.15
# Stratified in regard to Soc deviation, Volt deviation and both.
train, validation, test, test_final = [], [], [], []


def add_split_df(df):
    """ Splits the contents of a dataframe into 4 splits with
    0.5 train, 0.2 val, 0.15 test and 0.15 test_final.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing tinnumbers in column "tin_an"

    Returns
    -------
    None
        Modifies global lists train, validation, test and test_final.
    """
    global train, validation, test, test_final
    train_df, rest = train_test_split(df, test_size=0.5, random_state=0)
    train = train + (train_df["tin_an"].tolist())
    val_df, rest = train_test_split(rest, test_size=0.6, random_state=0)
    validation = validation + (val_df["tin_an"].tolist())
    test_df, test_final_df = train_test_split(rest, test_size=0.5, random_state=0)
    test = test + (test_df["tin_an"].tolist())
    test_final = test_final + (test_final_df["tin_an"].tolist())


# Add samples which fail (soc, volt and both)
add_split_df(df_tin_fail)
df_tin_fail_soc = df_tin_fail_soc.loc[
    ~(df_tin_fail_soc["tin_an"].isin(df_tin_fail["tin_an"])), :]
df_tin_fail_volt = df_tin_fail_volt.loc[
    ~(df_tin_fail_volt["tin_an"].isin(df_tin_fail["tin_an"])), :]
add_split_df(df_tin_fail_soc)
df_tin_fail_volt = df_tin_fail_volt.loc[
    ~(df_tin_fail_volt["tin_an"].isin(df_tin_fail_soc["tin_an"])), :]
add_split_df(df_tin_fail_volt)

# Add all healthy tin numbers
df_healthy = df_tin.loc[~(df_tin["tin_an"].isin(df_tin_fail["tin_an"])), :]
df_healthy = df_healthy.loc[~(df_healthy["tin_an"].isin(df_tin_fail_soc["tin_an"])), :]
df_healthy = df_healthy.loc[~(df_healthy["tin_an"].isin(df_tin_fail_volt["tin_an"])), :]
add_split_df(df_healthy)

# Create dictionary to store lists
dict_dataset_split = {"train": train, "validation": validation,
                      "val": validation, "test": test, "test_final": test_final}

# Save dictionary as pickle file
with open('dataset_split_dict.pickle', 'wb') as handle:
    pickle.dump(dict_dataset_split, handle, protocol=pickle.HIGHEST_PROTOCOL)
