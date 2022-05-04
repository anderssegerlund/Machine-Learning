# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: sohvida
#     language: python
#     name: sohvida
# ---

# # Exploratory data analysis

# ## Imports and loading data

# +
#Imports 

import pandas as pd
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from edec.read_data import FS, DataLake
import edec.preprocessing as pp
import collections

# # !pip install -e ".[dev]" Kolla upp detta senare

# +
# Loading meta data
# This will read env variable 
load_dotenv("../.env.dev")  # take environment variables from .env.

# ## Reading the data
#
# Now it is time to read the data.
# azure has a file system
fs = FS(os.environ["DATA_ACCOUNT"])


# ## Meta

# # +
# And we have a client to interact with the data
dl_meta = DataLake(
    os.environ["DATA_ACCOUNT"], # account
    os.environ["META_DATA_PATH"] # path
    )

# then we can read it as a pandas. OBS, data in parquet
df_meta = pd.read_parquet(
    dl_meta,
    filesystem=fs,
)


# And we have a client to interact with the data
dl = DataLake(
    os.environ["DATA_ACCOUNT"], # account
    os.environ["DATA_PATH"] # path
    )


# -

# Load the data and reformat it, size_data between 1 and 100, add_dev: True if you want Max Cell Deviations (SOC and Voltage) as columns, takes longer.
df = pp.load_data(dl, fs, size_data = 1, add_dev = False)

df = pd.read_parquet(
    dl[:1],
    filesystem=fs,
)

# ## Df_meta basic info

# Displaying Df_meta data
display(df_meta.columns)
display('Number of samples: '+str(df_meta.shape[0]))
df_meta.head()

# Counting number of uniques of each feature
display(df_meta.nunique())

# Checking distribution of samples with regard to features.
display(df_meta['Retail_Partner_Country_Code'].value_counts())
df_meta['Assembly_Plant_Description'].value_counts()

# Checking if there are missing data
if df_meta.isna().sum().sum() == 0:
    display("No missing data")

# Check date-range, range between 2020-03-27 and 2022-02-01
df_meta.sort_values(by="Factory_Complete_Date")

# ## Df basic info

# Displaying df-data
display(df.columns)
display('Number of samples: '+str(df.shape[0]))
display(df.head())

# Sort by time of readouts, ranging between 2020-11-11 and 2022-02-07
df.sort_values(by = "timestamp")

# ## Data Visualization

# Load the entire dataset in chunks and save all tin_an numbers 
all_tin = []
chunk_size = 4
for i in range(int(len(dl)/chunk_size)):
    df = pd.read_parquet(
        dl[i:i+chunk_size],
        filesystem=fs, columns=['tin_an']
    )
    for sample in df['tin_an']:
        all_tin.append(sample)


# Number of total samples
display('Total samples: '+str(len(all_tin)))

# Count occurences of each tin_an number and display in histogram
occurences = collections.Counter(all_tin)
test = collections.Counter(occurences).values()
hist_list = []
for a,b in collections.Counter(occurences).items():
    hist_list.append(b)
plt.hist(hist_list, bins=50)
plt.title('Distribution of # samples for each battery pack')
plt.xlabel('# Samples for a battery pack')
plt.ylabel('# Frequency')
plt.show()

# Load the entire dataset in chunks and save all F120/SW numbers
all_F120 = []
for i in range(100):
    df = pd.read_parquet(
        dl[i:i+1],
        filesystem=fs, columns=['F120/SW']
    )
    for sample in df['F120/SW']:
        all_F120.append(sample)

# Value count for F120/diagnostic part numbers.
srs = pd.Series(all_F120)
display(srs.value_counts())


# +
# Function for plotting cell values from a sample.

def plot_sample(sample, metric):
    data = sample[metric]
    fig = plt.figure(figsize=(16,6))
    plt.plot(data,'-o')
    plt.xlim(-5,113)
    plt.ylim(np.mean(data)-5,np.mean(data)+5)
    plt.xlabel('Cell number')
    plt.ylabel(metric)


# -

# Plotting a random sample
plot_sample(df.iloc[150],'Sorted_Voltage')
plt.title('Random sample')
plt.xlabel('Cell number')
plt.ylabel('Cell voltage')

# Plotting a random sample
plot_sample(df.iloc[350],'Sorted_SOC')
plt.title('Random sample')
plt.xlabel('Cell number')
plt.ylabel('Cell SOC')

# Plotting sample with largest cell voltage deviation
plot_sample(df.iloc[np.argmax(all_max_volt_dev)],'Sorted_Voltage')
plt.title('Sample with largest cell voltage deviation')
plt.ylabel('Cell Voltage')

# Histogram showing the distribution of max cell voltage deviation
plt.hist(df['Max_Cell_Deviation'], bins = 100)
plt.title('Distribution of the max cell voltage deviation')
plt.xlabel('Deviation')
plt.ylabel('# Samples')

# Check how many samples have a cell that deviates at least 1 % percent from the others (voltage)
temp_list = [item for item in df['Max_Cell_Deviation'] if item > 0.01]
count = len(temp_list)
perc_dev = 100*count/len(df['Max_Cell_Deviation'])
display(str(np.around(perc_dev,5))+'% of the samples deviate by more than 1 percent.')

# Histogram showing the distribution of max cell SOC deviation
plt.hist(df['Max_Cell_Deviation_SOC'], bins = 100)
plt.title('Distribution of the max cell SOC deviation')
plt.xlabel('Deviation')
plt.ylabel('# Samples')

# Check how many samples have a cell that deviates at least 1 % percent from the others (SOC)
temp_list = [item for item in df['Max_Cell_Deviation_SOC'] if item > 0.5]
count = len(temp_list)
perc_dev = 100*count/len(df['Max_Cell_Deviation_SOC'])
display(str(np.around(perc_dev,5))+'% of the samples deviate by more than 5 percent.')

# Add columns with mean voltage and mean SOC to the dataframe
df = pp.apply_function_df(df, 'Sorted_Voltage',np.mean,'Mean_Voltage')
df = pp.apply_function_df(df, 'Sorted_SOC',np.mean,'Mean_SOC')

# Histogram showing the distribution of the mean voltage
plt.hist(df['Mean_Voltage'], bins= 100)
plt.title('Distribution of mean voltage')
plt.xlabel('Mean voltage')
plt.ylabel('# Samples')

# Histogram showing the distribution of the mean SOC
plt.hist(df['Mean_SOC'], bins= 100)
plt.title('Distribution of mean SOC')
plt.xlabel('Mean SOC')
plt.ylabel('# Samples')

# Pairplot comparing the mean voltage and the mean SOC
plt.title('Pairplot comparing Mean Voltage and Mean SOC')
sns.scatterplot(df['Mean_Voltage'],df['Mean_SOC'])
plt.xlabel('Mean Voltage')
plt.ylabel('Mean SOC')
plt.show()

