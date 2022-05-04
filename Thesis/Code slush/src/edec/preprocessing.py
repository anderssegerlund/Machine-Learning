"""
========================================================================================
File with data processing functions
========================================================================================
"""
import numpy as np
import pandas as pd


def format_cell_voltage(volt_str: str):
    """ Converts voltage data string into list of tuples
    containing cell index and corresponding voltage value.

    Parameters
    ----------
    df: pd.DataFrame
        Column of Volt-series containing strings on the given format.

    Returns
    -------
    list of tuples
        returns a list of tuples where the first value is the cell index
        and the second one is the voltage value.
    """

    def helper_volt(single_volt_str: str):
        """ Helper function for format_cell_voltage.

        Parameters:
        -----------
        volt_str: str
            String in format eg. '4806/Cell_Voltage_048=3.8648,'

        Returns
        -------
        tuple of int and float:
            tuple with cell number (int) and float with the cell voltage value,
            eg. (48, 3.8648).
        """

        cell_indx, volt_val = single_volt_str.split('=')
        volt_val = volt_val.replace(',', '')
        volt_val = volt_val.replace('}', '')
        cell_indx = cell_indx.replace('4806/Cell_Voltage_', '')
        cell_indx = cell_indx.replace('{', '')
        volt_val = float(volt_val)
        cell_indx = int(cell_indx)

        return (cell_indx, volt_val)

    battery_cells = volt_str.split()
    battery_cells = list(map(helper_volt, battery_cells))
    voltage = np.zeros(len(battery_cells))
    for (a, b) in battery_cells:
        voltage[a - 1] = b
    return voltage


def format_cell_soc(soc_str: str):
    """ Converts SOC data string into list of tuples containing
    cell index and corresponding soc-value.

    Parameters
    ----------
    df: pd.Dataframe
        Column of SOC-series containing strings on the given format.

    Returns
    -------
    list of tuples
        List containing tuples with the first value being cell index
        and the second one being soc-value.
    """

    def helper_soc(single_soc_str: str):
        """ Helper function for format_cell_soc.

        Parameters
        ----------
        soc_str: str
            String in format eg. 'DA02/SOC_Percent_Cell_036=66.41,'

        Returns
        -------
        tuple of int and float
            Tuple with cell number (int) and float with the cell soc value,
            eg. (36, 66.41).
        """

        cell_indx, soc_val = single_soc_str.split('=')
        soc_val = soc_val.replace(',', '')
        soc_val = soc_val.replace('}', '')
        cell_indx = cell_indx.replace('DA02/SOC_Percent_Cell_', '')
        cell_indx = cell_indx.replace('{', '')
        soc_val = float(soc_val)
        cell_indx = int(cell_indx)
        return (cell_indx, soc_val)

    battery_cells = soc_str.split()
    battery_cells = list(map(helper_soc, battery_cells))
    soc = np.zeros(len(battery_cells))
    for (a, b) in battery_cells:
        soc[a - 1] = b
    return soc


def drop_empty_samples(df: pd.DataFrame):
    """ Finds and drops empty samples.
    Empty samples here are missing Voltage/SOC values (entered as {})

    Parameters
    ----------
    df: pd.Dataframe
        The dataframe from which samples should be found and dropped.

    Returns
    -------
    pd.Dataframe
        Modified dataframe without the empty samples.
    """

    df_cleaned = df[df['4806/Cell_Voltage'] != '{}']
    df_cleaned = df_cleaned[df_cleaned['DA02/SOC_Percent_Cell'] != '{}']
    return df_cleaned


def clean_data(df: pd.DataFrame):
    """ Removes empty samples and reformats Voltage and SOC data.

    Parameters
    ----------
    df: pd.Dataframe
        The dataframe that should be cleaned.

    Returns
    -------
    pd.Dataframe
        Modified dataframe, deleted empty samples, reformated features.
    """
    # Set index to rid number
    if 'rid' in df.columns:
        df.set_index('rid', inplace=True)

    # Remove empty samples.
    df_cleaned = drop_empty_samples(df)

    # Add features that hold voltage and soc as sorted lists.
    df_cleaned = apply_function_df(
        df_cleaned, "4806/Cell_Voltage", format_cell_voltage, "Sorted_Voltage")
    df_cleaned = apply_function_df(
        df_cleaned, 'DA02/SOC_Percent_Cell', format_cell_soc, "Sorted_SOC")

    # Remove samples that do not have 108 cells (PHEVs).
    df_cleaned = df_cleaned.loc[df_cleaned["Sorted_Voltage"].map(len) == 108]

    # Remove samples where all the Voltage values are 0.
    df_cleaned = df_cleaned.loc[df_cleaned["Sorted_Voltage"].map(sum) > 0.1]

    df_cleaned = df_cleaned.drop(
        ['4806/Cell_Voltage', 'DA02/SOC_Percent_Cell'], axis=1)

    # Remove samples which include negative SOC-values
    df_cleaned = df_cleaned.loc[df_cleaned["Sorted_SOC"].map(min) > 0.1]

    # Remove samples where the sum of all SOC values are smaller than 5 percent
    # df_cleaned = df_cleaned.loc[df_cleaned["Sorted_SOC"].map(min) > 5]

    return df_cleaned


def preprocess_data(df: pd.DataFrame):
    """ Preprocesses data, optimizes memory usage
        by downcasting and changing datatypes.

    Parameters
    ----------
    df: pd.Dataframe
        The dataframe that should be preprocessed.

    Returns
    -------
    pd.Dataframe
        Modified dataframe.
    """

    # Optimize the memory usage.
    if "F120/SW" in df.columns:
        df["F120/SW"] = df["F120/SW"].astype("category")
    if "tin_an" in df.columns:
        df["tin_an"] = df["tin_an"].astype("category")

    # Downcasting
    df["Sorted_Voltage"] = df["Sorted_Voltage"].apply(
        pd.to_numeric, downcast="float")
    df["Sorted_SOC"] = df["Sorted_SOC"].apply(
        pd.to_numeric, downcast="float")

    return df


def feat_engineering(df: pd.DataFrame, birth_dict: dict = None):
    """ Feature engineering, adding features that contain
        Max Cell Deviation both for Voltage and SOC.
    Parameters
    ----------
    df: pd.Dataframe
        The dataframe that should be modified.
    birth_dict: Dictionary
        Dictionary mapping tin to first sample timestamp.

    Returns
    -------
    pd.Dataframe
        Modified dataframe.
    """
    # Add column with birth date (from factory complete data)
    if birth_dict is not None:
        df['Birth'] = df['tin_an'].apply(
            lambda tin: birth_dict[tin])

        df['Age'] = df.apply(
            lambda row: int((pd.Timestamp(row['timestamp'])
                            - pd.Timestamp(row['Birth'])).total_seconds() / 60),
            axis=1)

    # Add max cell deviation as a column
    df = apply_function_df(
        df, "Sorted_Voltage", compute_max_cell_dev,
        'Volt_Dev')

    df = apply_function_df(
        df, "Sorted_SOC", compute_soc_deviation,
        'Soc_Dev')

    return df


def apply_function_df(df: pd.DataFrame, column: str, func, new_column: str):
    """Applies function to selected column and adds the result
    as a new column in the dataframe.
    Column should contain numpy array of data such as produced
    by reformat_data.
    Parameters
    ----------
    df: pd.Dataframe
        The dataframe that the function should be applied on.
    column: str
        The label of the column which we apply the function on.
    func
        The function to apply.
    new_column: str
        The label of the new column.

    Returns
    -------
    pd.Dataframe
        The modified dataframe containing the new columns.
    """

    df_copy = df.copy()
    df_copy[new_column] = df[column].apply(lambda x: func(x))
    return df_copy


def compute_max_cell_dev(arr: np.array):
    """ Computes the max cell deviation for a sample.
        Cell deviation is defined as the quotient between a cell value
        and the median value of the other cells.
    Parameters
    ----------
    arr: np.array of floats
        The sample for which the max cell deviation should be computed.

    Returns
    -------
    float
        The max cell deviation for a sample.
    """
    arr_copy = np.sort(np.copy(arr))
    low_median = arr_copy[int(len(arr) / 2) - 1]
    high_median = arr_copy[int(len(arr) / 2)]
    all_dev = []
    for i in range(len(arr)):
        if arr[i] < high_median:
            percent_dev = np.abs(1 - (arr[i] / high_median))
        else:
            percent_dev = np.abs(1 - (arr[i] / low_median))
        all_dev.append(percent_dev)
    return max(all_dev)


def compute_cell_dev_array(arr: np.array):
    """ Computes the max cell deviation for a sample.
        Cell deviation is defined as the quotient between a cell value
        and the median value of the other cells.
    Parameters
    ----------
    arr: np.array of floats
        The sample for which the max cell deviation should be computed.

    Returns
    -------
    float
        The max cell deviation for a sample.
    """
    arr_copy = np.sort(np.copy(arr))
    low_median = arr_copy[int(len(arr) / 2) - 1]
    high_median = arr_copy[int(len(arr) / 2)]
    all_dev = []
    for i in range(len(arr)):
        if arr[i] < high_median:
            percent_dev = np.abs(1 - (arr[i] / high_median))
        else:
            percent_dev = np.abs(1 - (arr[i] / low_median))
        all_dev.append(percent_dev)
    return all_dev


def approx_max_cell_dev(arr: np.array):
    """ Approximates the max cell deviation for a sample.
    Cell deviation here is the quotient between a cell value
    and the median value of the all cells (including the cell itself).
    Parameters
    ----------
    arr: np.array of floats
        Sample from which the max cell deviation is computed.

    Returns
    -------
    float
        The approximated max cell deviation for the input array.
    """
    med = np.median(arr)
    max_dev = 0
    for i in range(len(arr)):
        if (1 - (arr[i] / med)) > max_dev:
            max_dev = (1 - (arr[i] / med))
    return max_dev


def compute_soc_deviation(arr: np.array):
    """ Computes soc deviation which is defined as the difference
    between the max and min cell soc.
    Parameters
    ----------
    arr: np.array of floats
        Sample from which the soc deviation is computed.

    Returns
    -------
    float
        The soc deviation for the input array.
    """
    dev = arr.max() - arr.min()
    return dev


def label_data(df: pd.DataFrame, threshold_soc: float = 9,
               threshold_volt: float = 0.01):
    """ Adds a column with labels to the dataframe.
    The label is True if BOTH the soc and voltage deviations
    are above their corresponding thresholds. Otherwise the
    label is False.
    Parameters
    ----------
    df: pd.Dataframe
        Dataframe containing the dataset which we should
        add labels to.
    threshold_soc: float
        Value which the soc deviation needs to be larger than
        to be labeled True.
    threshold_volt: float
        Value which the voltage deviation needs to be larger than
        to be labeled True.

    Returns
    -------
    pd.Dataframe
        Dataframe with added column of labels.
    """
    df['Label'] = df.apply(
        lambda row: row['Volt_Dev'] >= threshold_volt
        and row['Soc_Dev'] >= threshold_soc, axis=1
    )
    return df
