# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3.9.10 64-bit (conda)
#     language: python
#     name: python3
# ---

# ## Imports

# %reset

import edec.afterprocessing as ap
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
import matplotlib.pyplot as plt
# %matplotlib inline

# ## Functions

# +
def import_labels(label=1, db="df.db",n_random=10):
        """
        This function import all cars from sql with stated label 0/1
        Example: Car "abc" and "bcd" has at some point been labeled with a Cell fail 1

        tin_an  Fail
        "abc"   0
        "abc"   1
        "abc"   0
        "bcd"   0
        "bcd"   0
        "bcd"   1
        """
        # Select all cell fails
        # limit the result to only x for non fail cars
        #if label == 0:
        #        lim = "LIMIT 3"
        #else:
        #        lim = ""

        ## SELECT ALL UNIQUE CARS WITH 0 / 1
        """
        First sub query select lim random number of tin numbers with label 
        Seccond query select timestamp, sorted soc etc from the sub_query
        
        """

        #if label == 0:
                #n_random = 100
        #if label == 1:
               # n_random = 1000
                
        sub_query  = f"SELECT DISTINCT(main_table.tin_an) FROM main_table \
                        INNER JOIN cell_fail \
                        ON main_table.rid = cell_fail.rid \
                        AND cell_fail.Fail = {label} \
                        ORDER BY RANDOM() \
                         LIMIT {n_random}"

        query = f"SELECT main_table.tin_an, [timestamp], [Sorted_SOC], cell_fail.Fail FROM main_table \
                INNER JOIN cell_fail \
                ON main_table.rid = cell_fail.rid \
                WHERE [tin_an] in ({sub_query})"

        df = ap.load_sql(query , db=db)

        # Change to Timestamp format
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Sort by timestamp
        df = df.sort_values(by=["timestamp"],ascending=True)

        # Print import result
        n_unique = len(df["tin_an"].unique())
        print(f"{n_unique} Unique cars with label {label} has been imported with {len(df)} subsamples")
        return df

def sample_to_tensor_x(df_tensor, sub_samples=3):
    x = torch.zeros(sub_samples,108) # 20 as we use 20 subsample cars
    i_from = len(df_tensor) - sub_samples
    i_to =  len(df_tensor)

    for tensor_i, df_i in enumerate(reversed(range(i_from, i_to))):
        carcell_voltage = df_tensor["Sorted_SOC"].iloc[df_i]     
        x[tensor_i] = torch.FloatTensor(carcell_voltage)
    return x


def create_dataset(db="df.db", parameters=None, normalize_data=False):
    # Output
    data = []
    run_details = {"Fail types loaded":0, 
                    "Samples removed":{
                                        0:0, 
                                        1:0,
                                        },
                    "Dropped tin":{
                                    0:[],
                                    1:[]},
                                    }

        
    # Paramters
    roll = parameters["roll"]
    sub_sample = parameters["Sub sample"]
    drop_sample = parameters["drop_sample"]+1 #Definierar 0 som ingen drop, men måste lägga till 1 pga att loopen f

    for fail in range(2): ##################################################################### ÄNDRA TILL 2
        run_details["Fail types loaded"] +=1
        print(f"Run {fail} started")
        # Create df with 0 and 1
        df = import_labels(label=fail, db=db, n_random = parameters["nRandom samples"][fail])
        cars = df["tin_an"].unique()
        print(f"{len(cars)} Cars in df")
        for i_car in range(len(cars)): # Change to 2

            # Get unique tin_number for fail / healthy
            df_tin = df[df["tin_an"] == cars[i_car]]
            df_tin = df_tin.reset_index()
            if fail == 1:
                max_index  = df_tin[df_tin["Fail"] == 1].index.values
                last_idx = max_index[-1]
                df_tensor = df_tin[:last_idx+1]
            else:
                last_idx = len(df_tin) +1 #-1
                df_tensor = df_tin[:last_idx+1]


            #for roll in range(1,5):
                #display(df_tensor[-roll-sub_sample:-roll])

            ### Create rolling
            # 1 should be standard
            last_idx_temp = last_idx
            for r in range(1,roll+1):
                #df_tensor_short = df_tensor[last_idx_temp-(parameters["Sub sample"]):last_idx_temp+1]
                df_tensor_short = df_tensor[-r-sub_sample:-r]
                last_idx_temp -= 1

                #if len(df_tensor_short) < parameters["Sub sample"]:
                    #   min_length = parameters["Sub sample"]
                    #   print(f"Skipping df. df_tensor length {len(df_tensor_short)}. (min lentht {min_length})")

                # Create database
                if (len(df_tensor_short) >= (sub_sample)) & (r >= (drop_sample)):
                #if (len(df_tensor_short) >= (sub_sample)) & (r > (drop_sample)):
                #if (len(df_tensor_short) >= (sub_sample)):
                    x = sample_to_tensor_x(df_tensor_short, sub_samples=sub_sample)
                    if normalize_data:
                        #x = normalize(x, p=2, dim = 1) # 1 or 0 
                        # Normalize by xi - min(x) / (max(x) - min(x))
                        min_i = x.min().item()
                        max_i = x.max().item()
                        x = (x-min_i)/(max_i - min_i)
                    y = fail
                    data.append((x,y))
                if (len(df_tensor_short) < (sub_sample)):
                    run_details["Samples removed"][fail] +=1
                    run_details["Dropped tin"][fail].append(df_tensor["tin_an"].unique()[0])
                    #continue
    print(run_details)
    return data

# +
#db = "df_10.db"
db = "df.db"

# Annotate data (ONLY RUN IF NEW ANNOTATION IS REQUIRED)
#ap.annotate_db(fail_type="SOC", db=db)
# -



# ## Basic data analysis

# +
query = f"SELECT DISTINCT(main_table.tin_an), cell_fail.Fail FROM main_table \
        INNER JOIN cell_fail \
        ON main_table.rid = cell_fail.rid"

df_count = ap.load_sql(query , db=db)
df_count.groupby("Fail").count()
# -





# ## Select parameters for data creation

"""Paramater values:

Sub_sample: How mmany timesamples backwards should be included (0 only give 1 sample, i.e [1,108])
drop_sample: How many samples should be dropped (0 for no drop). DROP SAMPLE MUST BE LARGER THAN ROLL
roll: How many timesteps back should we roll (1 for no rolling) 
"""
parameters = {"nRandom samples":{
                                0 : 20,
                                1 : 20,
                                },
            "Sub sample":20,
            "drop_sample":0, # Default 0
            "roll":5         # Default 1 (Must be larger or equal to subsample)
            }

data =create_dataset(db="df.db", parameters=parameters, normalize_data=True)

data





# ## Create dataset

train_size = int(len(data)*0.8)
test_size = len(data) - train_size
train_set, test_set = torch.utils.data.random_split(data,[train_size, test_size])
batch_size = 10
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,drop_last=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True,drop_last=True)

# ## Testa förhållandet 0 / 1 i train och test

val_0 = 0
val_1 = 0
for i in range(len(train_set)):
    if train_set[i][1] == 0:
        val_0 += 1
    if train_set[i][1] == 1:
        val_1 += 1
print((val_0)/(val_0+val_1))
print((val_1)/(val_0+val_1))


val_0 = 0
val_1 = 0
for i in range(len(test_set)):
    if test_set[i][1] == 0:
        val_0 += 1
    if test_set[i][1] == 1:
        val_1 += 1
print((val_0)/(val_0+val_1))
print((val_1)/(val_0+val_1))

for (X_train,y_train) in train_loader:
    break
X_train.shape




# ## Create network

subsample  = parameters["Sub sample"]
subsample



class ANNMultilayerperceptron(nn.Module):

    def __init__(self, input_size=(subsample*108),output_size=2, layers=[220,84]):  # 120, 84
        super().__init__()

        self.fc1 = nn.Linear(input_size, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc2b = nn.Linear(layers[1], 500)
        self.fc2c = nn.Linear(500, layers[1])
        self.fc2d = nn.Linear(layers[1], layers[1])
        self.fc3 = nn.Linear(layers[1], output_size)

    def forward(self,X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc2b(X))
        X = F.relu(self.fc2c(X))
        X = F.relu(self.fc2d(X))
        X = self.fc3(X)

        return F.log_softmax(X,dim=1) # PGA multiclass classification
        #return X


model = ANNMultilayerperceptron()
model



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#optimizer = torch.optim.Adadelta(model.parameters(), lr=0.001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# +
from ignite.metrics import Precision, Recall
from ignite.metrics import Precision ### LÄGG TILL IGNITE
train_precision = Precision()
train_recall = Recall()

test_precision = Precision()
test_recall = Recall()
# https://pytorch.org/ignite/metrics.html


epochs = 1000

train_losses = []
test_losses = []
train_correct = []
test_correct = []

# For loop epochs 

for i in range(epochs):
    trn_correct = 0
    tst_correct = 0 

    # Train

    for b, (X_train, y_train) in enumerate(train_loader):
        
        # Skip iteration if batch size not equal to stated dim
        
            
        #print(X_train.shape, y_train.shape) 
        
        b += 1
        
        y_pred = model(X_train.view(batch_size, -1))  # Flatten input
        lossTrain = criterion(y_pred, y_train)

        predicted = torch.max(y_pred.data,1)[1]

        #calculate precision and recall
        train_precision.update((y_pred, y_train))
        train_recall.update((y_pred, y_train))
      

        batch_corr = (predicted == y_train).sum()
        trn_correct += batch_corr

        optimizer.zero_grad()
        lossTrain.backward()
        optimizer.step()

        #if b%2 == 0:
           #print(f"Epoch {i} Batch: {b} Train Loss: {lossTrain.item()}")

    train_losses.append(lossTrain.data.item())
    train_correct.append(trn_correct)

    # Test
    with torch.no_grad():
        for b, (X_test,y_test) in enumerate(test_loader):
            y_val = model(X_train.view(batch_size, -1))

            predicted = torch.max(y_val.data,1)[1]

            #calculate precision and recall
            test_precision.update((y_val, y_test))
            test_recall.update((y_val, y_test))
            
            loss = criterion(y_val, y_test)
            test_losses.append(loss)
            test_correct.append(trn_correct)

        if b%5 == 0:
            print(f"Epoch {i} Batch: {b} Train Loss: {lossTrain.item()} Validation Loss: {loss.item()}")

# -



print(train_precision.compute())
print(train_recall.compute())
print(test_precision.compute())
print(test_recall.compute())



# +

plt.plot(train_losses, label="Train losses")
#plt.plot(test_losses, label= "Test losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
# -

# ## Predict a state of a single sample from test_set

import numpy as np
for _ in range(100):
    # Select random sample
    i = np.random.randint(1,len(test_set))
    x = test_set[i][0]
    y = test_set[i][1]


    # Evaluate on sample
    model.eval()
    with torch.no_grad():
        new_pred =model(x.view(1,-1))
        pred_int = int(torch.max(new_pred.data,1)[1])
    print(f"Random sample {i} selected with state {y}. Model predict state is {pred_int}")

# ## Predict a state of a single sample from train_set

import numpy as np
for _ in range(100):
    # Select random sample
    i = np.random.randint(1,len(train_set))
    x = train_set[i][0]
    y = train_set[i][1]


    # Evaluate on sample
    model.eval()
    with torch.no_grad():
        new_pred =model(x.view(1,-1))
        pred_int = int(torch.max(new_pred.data,1)[1])
    print(f"Random sample {i} selected with state {y}. Model predict state is {pred_int}")






