"""
========================================================================================
File with data afterprocessing functions
========================================================================================
"""
import sqlite3
import pandas as pd
import numpy as np
import torch
import edec.preprocessing as pp
import pickle


def str_tolist(string):
    """ Function for converting str to list.

    Parameters
    ----------
    string: str
        String in format of '[float float]'

    Returns
        Regular list
    """
    string = string.strip("[").strip("]").strip("\n")
    string = string.split(" ")

    for i in reversed(range(len(string))):
        if string[i] == "":
            string.remove(string[i])

    return [float(i) for i in string]


def load_sql(sql_command, db="df.db"):

    """ Function to load SQLite3 database to pandas dataframe
    """

    # Connection to db
    link = "../scripts/" + db
    conn = sqlite3.connect(link)

    # Load from sql
    df = pd.read_sql(sql_command, conn)
    conn.close()

    if "Sorted_Voltage" in df.columns:
        df["Sorted_Voltage"] = df["Sorted_Voltage"].apply(
            lambda i: str_tolist(i))

    if "Sorted_SOC" in df.columns:
        df["Sorted_SOC"] = df["Sorted_SOC"].apply(
            lambda i: str_tolist(i))

    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].apply(
            lambda i: pd.Timestamp(i))

    if "Birth" in df.columns:
        df["Birth"] = df["timestamp"].apply(
            lambda i: pd.Timestamp(i))

    if "Sorted_Voltage" in df.columns:
        df["Sorted_Voltage"] = df["Sorted_Voltage"].apply(
            pd.to_numeric, downcast="float")

    if "Sorted_SOC" in df.columns:
        df["Sorted_SOC"] = df["Sorted_SOC"].apply(
            pd.to_numeric, downcast="float")

    return df


def annotate_db(sql_db_name: str, threshold_soc=9.0, threshold_volt=0.01):
    """ Annotates data in sql according to input thresholds.
    Creates a table called "cell_fail" which contains labels
    namely "Fail", "Fail_Soc" and "Fail_Volt".

    Parameters
    ----------
    sql_db_name: str
        Filename of sql database where the dataframe is kept.
    threshold_soc: float
        Soc threshold, samples with higher Soc deviation than this
        threshold will be classified as "Fail_Soc".
    threshold_volt: float
        Voltage threshold, samples with higher volt deviation than this
        threshold will be classified as "Fail_Volt"

    Returns
        Nothing, updates "cell_fail"-table with columns with labels.
    """
    sql = "SELECT [rid],[Volt_dev],[Soc_Dev] FROM main_table"
    df = load_sql(sql, db=sql_db_name)

    # Label fails
    df["Fail"] = np.where(pp.label_data(
        df, threshold_soc, threshold_volt)["Label"], 1, 0)
    df["Fail_Soc"] = np.where(pp.label_data(df, threshold_soc, 0)["Label"], 1, 0)
    df["Fail_Volt"] = np.where(pp.label_data(df, 0, threshold_volt)["Label"], 1, 0)

    # Connect to SQL
    conn = sqlite3.connect("../scripts/" + sql_db_name)
    df.to_sql("cell_fail", con=conn, if_exists='replace')
    conn.close()


def import_labels(label=1, db="df.db", n_random=10, tin_exclude_list=None):
    """
    This function import all cars from sql with stated label 0/1
    Example: Car "abc" and "bcd" has at some point been labeled
    with a Cell fail 1

    tin_an  Fail
    "abc"   0
    "abc"   1
    "abc"   0
    "bcd"   0
    "bcd"   0
    "bcd"   1
    """

    """
    First sub query select lim random number of tin numbers with label
    Seccond query select timestamp, sorted soc etc from the sub_query

    tin_exclude_list can be added to exclude tin numbers from df
    must contain 2 elements and be in the form ('tin1', 'tin2')
    """
    if type(tin_exclude_list) is tuple:
        sub_sub_query = f"AND [tin_an] NOT IN {tin_exclude_list}"
    else:
        sub_sub_query = ""

    sub_query = f"SELECT DISTINCT(main_table.tin_an) FROM main_table \
                    INNER JOIN cell_fail \
                    ON main_table.rid = cell_fail.rid \
                    AND cell_fail.Fail = {label} \
                    ORDER BY RANDOM() \
                        LIMIT {n_random}"

    query = f"SELECT main_table.tin_an, [timestamp], [Sorted_SOC],[Sorted_Voltage],\
            cell_fail.Fail FROM main_table \
            INNER JOIN cell_fail \
            ON main_table.rid = cell_fail.rid \
            WHERE [tin_an] in ({sub_query}) {sub_sub_query}"

    df = load_sql(query, db=db)

    # Change to Timestamp format
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Sort by timestamp
    df = df.sort_values(by=["timestamp"], ascending=True)

    # Print import result
    n_unique = len(df["tin_an"].unique())
    print(f"{n_unique} Unique cars with label {label} has",
          f"been imported with {len(df)} subsamples")
    return df


def sample_to_tensor_x(df_tensor, sub_samples=3):
    """
    Helper function for create_dataset
    """
    x = torch.zeros(sub_samples, 108)
    i_from = len(df_tensor) - sub_samples
    i_to = len(df_tensor)

    for tensor_i, df_i in enumerate(reversed(range(i_from, i_to))):
        carcell_voltage = df_tensor["Sorted_SOC"].iloc[df_i]
        x[tensor_i] = torch.FloatTensor(carcell_voltage)
    return x


def create_dataset(db="df.db", parameters=None, normalize_data=False,
                   show_rundetails=False):
    # Output
    data = []
    run_details = {"Fail types loaded": 0,
                   "Samples removed": {0: 0,
                                       1: 0,
                                       },
                   "Dropped tin": {0: [],
                                   1: [],
                                   },
                   }

    # Paramters
    roll = parameters["roll"]
    sub_sample = parameters["Sub sample"]
    drop_sample = parameters["drop_sample"] + 1

    for fail in range(2):
        run_details["Fail types loaded"] += 1
        print(f"Run {fail} started")
        # Create df with 0 and 1
        df = import_labels(label=fail, db=db, n_random=parameters[
            "nRandom samples"][fail])

        cars = df["tin_an"].unique()
        print(f"{len(cars)} Cars in df")

        for i_car in range(len(cars)):

            # Get unique tin_number for fail / healthy
            df_tin = df[df["tin_an"] == cars[i_car]]
            df_tin = df_tin.reset_index()

            if fail == 1:
                max_index = df_tin[df_tin["Fail"] == 1].index.values
                last_idx = max_index[-1]
                df_tensor = df_tin[:last_idx + 1]
            else:
                last_idx = len(df_tin) + 1
                df_tensor = df_tin[:last_idx + 1]

            # Create rolling
            last_idx_temp = last_idx
            for r in range(1, roll + 1):

                if r == 1:
                    r_end = None
                else:
                    r_end = -r + 1

                df_tensor_short = df_tensor[- r - sub_sample + 1:r_end]
                last_idx_temp -= 1

                # Create database
                if (len(df_tensor_short) >= (
                   sub_sample)) & (r >= (drop_sample)):
                    x = sample_to_tensor_x(
                        df_tensor_short, sub_samples=sub_sample)

                    if normalize_data:
                        # Normalize by xi - min(x) / (max(x) - min(x))
                        min_i = x.min().item()
                        max_i = x.max().item()
                        x = (x - min_i) / (max_i - min_i)
                    y = fail
                    data.append((x, y))
                if (len(df_tensor_short) < (sub_sample)):
                    run_details["Samples removed"][fail] += 1
                    run_details["Dropped tin"][fail].append(
                        df_tensor["tin_an"].unique()[0])

        if show_rundetails:
            print(run_details)
    return data


def import_labels_tin(label=1, db="df.db", tin_list=None):
    """
    This function import all cars from sql with stated label 0/1
    Example: Car "abc" and "bcd" has at some point been labeled
    with a Cell fail 1

    tin_an  Fail
    "abc"   0
    "abc"   1
    "abc"   0
    "bcd"   0
    "bcd"   0
    "bcd"   1
    """

    """
    First sub query select lim random number of tin numbers with label
    Seccond query select timestamp, sorted soc etc from the sub_query
    """
    # Transform tin_list
    tin_list = str(tin_list).strip("[").strip("]")

    query = f"SELECT main_table.tin_an, [timestamp], [Sorted_SOC],\
            cell_fail.Fail FROM main_table \
            INNER JOIN cell_fail \
            ON main_table.rid = cell_fail.rid \
            AND cell_fail.Fail = {label} \
            WHERE [tin_an] in ({tin_list})"

    df = load_sql(query, db=db)

    # Change to Timestamp format
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Sort by timestamp
    df = df.sort_values(by=["timestamp"], ascending=True)

    # Print import result
    n_unique = len(df["tin_an"].unique())
    print(f"{n_unique} Unique cars with label {label} has",
          f"been imported with {len(df)} subsamples")
    return df


def create_tin_dataset(db="df.db", parameters=None, normalize_data=False,
                       show_rundetails=False, tin_list=None):
    # Output
    data = []
    run_details = {"Fail types loaded": 0,
                   "Samples removed": {0: 0,
                                       1: 0,
                                       },
                   "Dropped tin": {0: [],
                                   1: [],
                                   },
                   }

    # Paramters
    roll = parameters["roll"]
    sub_sample = parameters["Sub sample"]
    drop_sample = parameters["drop_sample"] + 1

    for fail in range(2):
        run_details["Fail types loaded"] += 1
        print(f"Run {fail} started")
        # Create df with 0 and 1
        df = import_labels_tin(label=fail, db=db, tin_list=tin_list)

        cars = df["tin_an"].unique()
        print(f"{len(cars)} Cars in df")

        for i_car in range(len(cars)):

            # Get unique tin_number for fail / healthy
            df_tin = df[df["tin_an"] == cars[i_car]]
            df_tin = df_tin.reset_index()

            if fail == 1:
                max_index = df_tin[df_tin["Fail"] == 1].index.values
                last_idx = max_index[-1]
                df_tensor = df_tin[:last_idx + 1]
            else:
                last_idx = len(df_tin) + 1
                df_tensor = df_tin[:last_idx + 1]

            # Create rolling
            last_idx_temp = last_idx
            for r in range(1, roll + 1):

                if r == 1:
                    r_end = None
                else:
                    r_end = -r + 1

                df_tensor_short = df_tensor[- r - sub_sample + 1:r_end]
                last_idx_temp -= 1

                # Create database
                if (len(df_tensor_short) >= (
                   sub_sample)) & (r >= (drop_sample)):
                    x = sample_to_tensor_x(
                        df_tensor_short, sub_samples=sub_sample)

                    if normalize_data:
                        # Normalize by xi - min(x) / (max(x) - min(x))
                        min_i = x.min().item()
                        max_i = x.max().item()
                        x = (x - min_i) / (max_i - min_i)
                    y = fail
                    data.append((x, y))
                if (len(df_tensor_short) < (sub_sample)):
                    run_details["Samples removed"][fail] += 1
                    run_details["Dropped tin"][fail].append(
                        df_tensor["tin_an"].unique()[0])

        if show_rundetails:
            print(run_details)
    return data


def create_sequences_lstm(tins: list, faulty: int, features_single: list = None,
                          features_nest: list = None, n_history: int = 5):
    """ Creates data sequences in format that is suitable
    for lstm input.

    Parameters
    ----------
    tins: list of str
        List containing tin numbers that samples should be created from.
    faulty: bool
        Are the tin numbers faulty or not, True/1 if faulty.
    features_single: list of str
        The single features that should be included (eg. Soc_Dev)
    features_nest: list of str
        The nested features that should be included (eg. Sorted_Volt)
    n_history: int
        How many time steps should each sequence contain?

    Returns
        List with samples. Each sample is a tuple with the first
        element being a tensor of size n_history*n_features.
    """

    sequences = []
    tins = tuple(tins)
    query = f'SELECT tin_an, Sorted_Voltage, Sorted_SOC, main_table.Volt_Dev, main_table.Soc_Dev, \
    Age, main_table.rid, Label FROM main_table \
    INNER JOIN cell_fail \
    ON main_table.rid = cell_fail.rid \
    WHERE tin_an in {tins} ORDER BY Age DESC'
    df_temp = load_sql(query, db="df.db")
    for tin_an, group in df_temp.groupby("tin_an"):
        found = False
        for i in range(len(group) - n_history):
            lab = group.iloc[i]['Label']
            if faulty and lab == 0 and not found:
                continue
            if faulty and lab == 1:
                found = True
            if features_single is not None:
                sample1 = group[features_single][i:i + n_history].to_numpy()
            if features_nest is not None:
                sample2 = group[features_nest][i:i + n_history]
                sample2 = sample2.to_numpy()
                sample2 = np.hstack(sample2)
                sample2 = np.hstack(sample2).reshape(n_history, -1)
            if features_single is not None and features_nest is not None:
                sample = np.concatenate((sample1, sample2), axis=1)
                sample_tensor = torch.tensor(sample)
            elif features_single is None:
                sample_tensor = torch.tensor(sample2)
            elif features_nest is None:
                sample_tensor = torch.tensor(sample1)
            sequences.append((sample_tensor, int(faulty)))
    return sequences


def get_subset_data(dataset: str = "train", n_healthy: int = 44,
                    n_failed: int = 44, separate: bool = False):
    """ Retrieves a list with specified amount of healthy
    and failed tin numbers from the training set.

    Parameters
    ----------
    dataset: str
        Which dataset split you want to retrieve samples from,
        can be train, val, test, test_final.
    n_healthy: int
        Amount of healthy samples that we want to retrieve.
    n_failed: int
        Amount of failed samples that we want to retrieve.

    Returns
    -------
    list of strings
        list containing the sought tin numbers
    """

    li = []
    with open('../scripts/dataset_split_dict.pickle', 'rb') as handle:
        datasplit_dict = pickle.load(handle)

    all_tin = datasplit_dict[dataset]

    query = 'SELECT DISTINCT main_table.tin_an FROM main_table \
    INNER JOIN cell_fail \
    ON main_table.rid = cell_fail.rid \
    WHERE cell_fail.Fail == 1'
    df_tin_fail = load_sql(query, db="df.db")
    tin_fail = df_tin_fail["tin_an"].tolist()

    fail = [x for x in all_tin if x in tin_fail]
    li = li + fail[:n_failed]
    healthy = [x for x in all_tin if x not in tin_fail]
    li = li + healthy[:n_healthy]
    if separate:
        return fail[:n_failed], healthy[:n_healthy]
    return li


def get_subset_data_claim(dataset: str = "train", n_healthy: int = 44,
                          n_failed: int = 44, separate: bool = False):
    """ Retrieves a list with specified amount of healthy
    and failed tin numbers from the training set.

    Parameters
    ----------
    dataset: str
        Which dataset split you want to retrieve samples from,
        can be train, val, test, test_final.
    n_healthy: int
        Amount of healthy samples that we want to retrieve.
    n_failed: int
        Amount of failed samples that we want to retrieve.

    Returns
    -------
    list of strings
        list containing the sought tin numbers
    """

    li = []
    with open('../scripts/dataset_split_dict_claim_all.pickle', 'rb') as handle:
        failed, datasplit_dict = pickle.load(handle)
    all_tin = datasplit_dict[dataset]
    fail = [x for x in failed if x in all_tin]
    li = li + fail[:n_failed]
    healthy = [x for x in all_tin if x not in failed]
    li = li + healthy[:n_healthy]
    if separate:
        return fail[:n_failed], healthy[:n_healthy]
    return li


def create_sequences_lstm_claim(tins: list, faulty: int, features_single: list = None,
                                features_nest: list = None, n_history: int = 5):
    """ Creates data sequences in format that is suitable
    for lstm input.

    Parameters
    ----------
    tins: list of str
        List containing tin numbers that samples should be created from.
    faulty: bool
        Are the tin numbers faulty or not, True/1 if faulty.
    features_single: list of str
        The single features that should be included (eg. Soc_Dev)
    features_nest: list of str
        The nested features that should be included (eg. Sorted_Volt)
    n_history: int
        How many time steps should each sequence contain?

    Returns
        List with samples. Each sample is a tuple with the first
        element being a tensor of size n_history*n_features.
    """

    sequences = []
    tins = tuple(tins)
    query = f'SELECT main_table.tin_an, Sorted_Voltage, Sorted_SOC, main_table.Volt_Dev, \
    main_table.Soc_Dev, Age, main_table.rid, cell_fail_claim.Label FROM main_table \
    INNER JOIN cell_fail_claim \
    ON main_table.rid = cell_fail_claim.rid \
    WHERE main_table.tin_an in {tins} ORDER BY Age DESC'
    df_temp = load_sql(query, db="df.db")
    for tin_an, group in df_temp.groupby("tin_an"):
        found = False
        for i in range(len(group) - n_history):
            lab = group.iloc[i]['Label']
            if faulty and lab == 0 and not found:
                continue
            if faulty and lab == 1:
                found = True
            if features_single is not None:
                sample1 = group[features_single][i:i + n_history].to_numpy()
            if features_nest is not None:
                sample2 = group[features_nest][i:i + n_history]
                sample2 = sample2.to_numpy()
                sample2 = np.hstack(sample2)
                sample2 = np.hstack(sample2).reshape(n_history, -1)
            if features_single is not None and features_nest is not None:
                sample = np.concatenate((sample1, sample2), axis=1)
                sample_tensor = torch.tensor(sample)
            elif features_single is None:
                sample_tensor = torch.tensor(sample2)
            elif features_nest is None:
                sample_tensor = torch.tensor(sample1)
            sequences.append((sample_tensor, int(faulty)))
    return sequences
