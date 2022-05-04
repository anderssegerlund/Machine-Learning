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

import edec.afterprocessing as ap
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pickle


# # Different examles how data can be created

# ### 1. Data creation for ANN 
# This function return a data set of: 
# - n random tin numbers with label 0 
# - m random tin numbers with label 1  
# - Rolling, sample and drop can be set in parameters

current_db = "df.db"
# Annotate data if needed
# ap.annotate_db(fail_type="SOC", db=current_db)

"""Paramater values:

Sub_sample: How mmany timesamples backwards should be included (0 only give 1 sample, i.e [1,108])
drop_sample: How many samples should be dropped (0 for no drop). DROP SAMPLE MUST BE LARGER THAN ROLL
roll: How many timesteps back should we roll (1 for no rolling) 
"""
parameters = {"nRandom samples": {
                                0: 20,
                                1: 20,
                                },
              "Sub sample": 5,
              "drop_sample": 0,  # Default 0
              "roll": 1          # Default 1 (Must be larger or equal to subsample)
              }

data = ap.create_dataset(db="df.db", parameters=parameters,
                            normalize_data=True)

# +
train_size = int(len(data)*0.8)
test_size = len(data) - train_size
train_set, test_set = torch.utils.data.random_split(data,[train_size, test_size])
batch_size = 10

train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                          shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size,
                         shuffle=True, drop_last=True)
# -

subsample = parameters["Sub sample"]
subsample


# ### Test of model data in ANN network

# +
class ANNMultilayerperceptron(nn.Module):

    def __init__(self, input_size=(subsample*108),output_size=2, layers=[220, 84]):  # 120, 84
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

        return F.log_softmax(X, dim=1) # PGA multiclass classification
        #return X
model = ANNMultilayerperceptron()

for b, (X_train, y_train) in enumerate(train_loader):
    break

model(X_train.view(batch_size, -1))
# -



# ## 2. ANN samma som ovan men med tin imput

with open('../scripts/dataset_split_dict.pickle', 'rb') as handle:
    datasplit_dict = pickle.load(handle)
tr = datasplit_dict["train"]

# +
#train_tin = datasplit_dict["train"] # Select all train tin
#train_tin = ap.get_subset_traindata(n_failed=44, n_healthy=10)  #Select defined amount
#train_tin_fail, train_tin_healthy = ap.get_subset_data(dataset="train", n_healthy=44, n_failed=44, separate=True)
train_tin = ap.get_subset_data(dataset="train", n_healthy=44, n_failed=44, separate=False)


validation_tin = datasplit_dict["validation"]
test_tin = datasplit_dict["test"]
test_final_tin = datasplit_dict["test_final"]

# +
train_parameters = {
            "Sub sample": 2,
            "drop_sample": 0,  # Default 0
            "roll": 1          # Default 1 (Must be smaller or equal to subsample)
            }

test_parameters = {
            "Sub sample": 2,
            "drop_sample": 0,  # Default 0
            "roll": 1          # Default 1 (Must be smaller or equal to subsample)
            }

train_set = ap.create_tin_dataset(db="df.db", parameters=train_parameters, normalize_data=True,
                   show_rundetails=False, tin_list=train_tin)

test_set = ap.create_tin_dataset(db="df.db", parameters=test_parameters, normalize_data=True,
                   show_rundetails=False, tin_list=test_tin)
# -

batch_size = 10
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,drop_last=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True,drop_last=True)

subsample = train_parameters["Sub sample"]
subsample


# ### Test of model data in ANN network

# +
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

for b, (X_train, y_train) in enumerate(train_loader):
    break

model(X_train.view(batch_size, -1))
# -






