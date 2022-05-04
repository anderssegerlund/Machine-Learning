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

# + [markdown] tags=[]
# ## Import and create connection to sql
# -

import sqlite3
import pandas as pd
import numpy as np
import edec.afterprocessing as ap
import edec.preprocessing as pp
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader, Dataset
from sklearn import preprocessing
from torchvision import transforms

# Create connection to sql
link = "../scripts/data_sql"
conn = sqlite3.connect(link)

# + [markdown] tags=[]
# ## All tin numbers and tables
# -

query = 'SELECT DISTINCT tin_an FROM main_table'
df_tin = ap.load_sql(query, db = "data_sql")

con = sqlite3.connect('../scripts/data_sql')
cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())


# + [markdown] tags=[]
# ## Create input function and annotate data

# +
def create_input(tin: str, faulty: int, n_history: int = 5, sql_path: str = "../scripts/data_sql"):
    conn = sqlite3.connect(sql_path)
    query = 'SELECT Sorted_Voltage, Sorted_SOC, main_table.Volt_Dev, main_table.Soc_Dev, \
    Age, main_table.rid, Label FROM main_table \
    INNER JOIN cell_fail \
    ON main_table.rid = cell_fail.rid \
    WHERE tin_an == "%s" ORDER BY Age DESC' % tin
    df_temp = ap.load_sql(query, db = "data_sql")
    df_new = pd.DataFrame(columns=['Input','Label'])
    found = False
    for i in range(len(df_temp)-n_history):
        lab = df_temp.iloc[i]['Label']
        if faulty and lab == 0 and not found:
            continue
        if faulty and lab == 1:
            found = True
        sample = np.empty(0)
        for j in range(i,i+n_history):
            row = df_temp.iloc[j]["Sorted_Voltage"]
            row = np.concatenate((row, df_temp.iloc[j]["Sorted_SOC"]))
            np_values = df_temp.loc[j,['Volt_Dev','Soc_Dev','Age']]
            row = np.append(row, np_values)
            sample = np.append(sample, row)
        df_new.loc[len(df_new)] = [sample, faulty]
    return df_new

def create_input_arr(tin_arr, faulty):
    df = pd.DataFrame(columns=['Input', 'Label'])
    for i in range(len(tin_arr)):
        tin = tin_arr[i]
        frames = [df, create_input(tin, faulty)]
        df = pd.concat(frames, ignore_index=True)
        print(f'{i+1}/{len(tin_arr)} cars added')
    return df


# +
def create_input_only_dev(tin: str, faulty: int, n_history: int = 5, sql_path: str = "../scripts/data_sql"):
    conn = sqlite3.connect(sql_path)
    query = 'SELECT main_table.Volt_Dev, main_table.Soc_Dev, \
    main_table.rid, Label FROM main_table \
    INNER JOIN cell_fail \
    ON main_table.rid = cell_fail.rid \
    WHERE tin_an == "%s" ORDER BY Age DESC' % tin
    df_temp = ap.load_sql(query, db = "data_sql")
    df_new = pd.DataFrame(columns=['Input','Label'])
    found = False
    for i in range(len(df_temp)-n_history):
        lab = df_temp.iloc[i]['Label']
        if faulty and lab == 0 and not found:
            continue
        if faulty and lab == 1:
            found = True
        sample = np.empty(0)
        for j in range(i,i+n_history):
            np_values = df_temp.loc[j,['Volt_Dev','Soc_Dev']]
            sample = np.append(sample, np_values)
        df_new.loc[len(df_new)] = [sample, faulty]
    return df_new

def create_input_arr_dev(tin_arr, faulty):
    df = pd.DataFrame(columns=['Input', 'Label'])
    for i in range(len(tin_arr)):
        tin = tin_arr[i]
        frames = [df, create_input_only_dev(tin, faulty)]
        df = pd.concat(frames, ignore_index=True)
        print(f'{i+1}/{len(tin_arr)} cars added')
    return df


# +
def create_input_only_soc(tin: str, faulty: int, n_history: int = 5, sql_path: str = "../scripts/data_sql"):
    conn = sqlite3.connect(sql_path)
    query = 'SELECT main_table.Sorted_SOC, \
    main_table.rid, Label FROM main_table \
    INNER JOIN cell_fail \
    ON main_table.rid = cell_fail.rid \
    WHERE tin_an == "%s" ORDER BY Age DESC' % tin
    df_temp = ap.load_sql(query, db = "data_sql")
    df_new = pd.DataFrame(columns=['Input','Label'])
    found = False
    for i in range(len(df_temp)-n_history):
        lab = df_temp.iloc[i]['Label']
        if faulty and lab == 0 and not found:
            continue
        if faulty and lab == 1:
            found = True
        sample = np.empty(0)
        for j in range(i,i+n_history):
            row = df_temp.iloc[j]["Sorted_SOC"]
            sample = np.append(sample, row)
        df_new.loc[len(df_new)] = [sample, faulty]
    return df_new

def create_input_arr_soc(tin_arr, faulty):
    df = pd.DataFrame(columns=['Input', 'Label'])
    for i in range(len(tin_arr)):
        tin = tin_arr[i]
        frames = [df, create_input_only_soc(tin, faulty)]
        df = pd.concat(frames, ignore_index=True)
        print(f'{i+1}/{len(tin_arr)} cars added')
    return df


# -

# Labeling data, results end up in cell_fail table in column 'Fail'
ap.annotate_db(sql_db_name = 'data_sql', threshold_soc = 9, threshold_volt = 0.01)

# + [markdown] tags=[]
# ## Take out failed samples and failed tin numbers
# -

conn = sqlite3.connect("../scripts/data_sql")
query = 'SELECT DISTINCT main_table.tin_an FROM main_table \
INNER JOIN cell_fail \
ON main_table.rid = cell_fail.rid \
WHERE cell_fail.Fail == 1'
df_tin_fail = ap.load_sql(query, db = "data_sql")

conn = sqlite3.connect("../scripts/data_sql")
query = 'SELECT main_table.tin_an, cell_fail.Fail FROM main_table \
INNER JOIN cell_fail \
ON main_table.rid = cell_fail.rid \
WHERE cell_fail.Fail == 1'
df_failed_samples = ap.load_sql(query, db = "data_sql")

print('Number of failed samples: ', len(df_failed_samples))
print('Number of unique tin_an with failed sample: ', len(df_tin_fail))

# + [markdown] tags=[]
# ## Analysis/checks
# -

tin = df_tin_fail.iloc[0][0]
query = 'SELECT main_table.tin_an, main_table.Volt_Dev, main_table.Soc_Dev, Age, main_table.rid, Fail FROM main_table \
INNER JOIN cell_fail \
ON main_table.rid = cell_fail.rid \
WHERE tin_an == "%s" ORDER BY Age DESC' % tin
df_check = ap.load_sql(query, db = "data_sql")
print(len(df_check))
df_check[df_check['Fail'] == 1]

# Plot the soc deviation for a battery
soc = df_check['Soc_Dev'].to_numpy()
age = df_check['Age'].to_numpy()
soc = np.flip(soc)
age = np.flip(age)
fig, ax = plt.subplots()
ax.plot(age, soc)
ax.set_title('Soc deviation over time for failed battery')
ax.set_xlabel('Age [min]')
ax.set_ylabel('Soc Deviation [%]')
plt.show()

# + [markdown] tags=[]
# ## Create sets
# -

# Arrays with failed and healthy tin numbers
tin_fail = df_tin_fail.values.flatten()
df_tin_healthy = df_tin[~df_tin['tin_an'].isin(tin_fail)]
tin_healthy = df_tin_healthy.values.flatten()

# +
# Sample a number of random healthy tin numbers
tin_healthy_random = random.sample(list(tin_healthy), k=len(tin_fail))

x_train_fail, x_rest = train_test_split(tin_fail, test_size=0.4, random_state=0)
x_val_fail, x_rest = train_test_split(x_rest, test_size=0.6666, random_state=0)
x_test_fail, x_test_final_fail = train_test_split(x_rest, test_size=0.5, random_state=0)

x_train_healthy, x_rest = train_test_split(tin_healthy_random, test_size=0.4, random_state=0)
x_val_healthy, x_rest = train_test_split(x_rest, test_size=0.6666, random_state=0)
x_test_healthy, x_test_final_healthy = train_test_split(x_rest, test_size=0.5, random_state=0)
# -

print(len(x_train_fail), len(x_val_fail), len(x_test_fail), len(x_test_final_fail))
print(len(x_train_healthy), len(x_val_healthy), len(x_test_healthy), len(x_test_final_healthy))

# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ### Soc, Voltage, Soc-dev, Volt-dev and age
# -

# Create training set
frames = [create_input_arr(x_train_fail, faulty=1), create_input_arr(x_train_healthy, faulty=0)]
df_train = pd.concat(frames, ignore_index=True)

# Create validation set
frames = [create_input_arr(x_val_fail, faulty=1), create_input_arr(x_val_healthy, faulty=0)]
df_val = pd.concat(frames, ignore_index=True)

# Create test set
frames = [create_input_arr(x_test_fail, faulty=1), create_input_arr(x_test_healthy, faulty=0)]
df_test = pd.concat(frames, ignore_index=True)

# Create final test set
frames = [create_input_arr(x_test_final_fail, faulty=1), create_input_arr(x_test_final_healthy, faulty=0)]
df_test_final = pd.concat(frames, ignore_index=True)

# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ### Only deviations
# -

# Create training set only deviations
frames = [create_input_arr_dev(x_train_fail, faulty=1), create_input_arr_dev(x_train_healthy, faulty=0)]
df_train_dev = pd.concat(frames, ignore_index=True)

# Create validation set only deviation
frames = [create_input_arr_dev(x_val_fail, faulty=1), create_input_arr_dev(x_val_healthy, faulty=0)]
df_val_dev = pd.concat(frames, ignore_index=True)

# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ### Only soc values
# -

# Create training set only deviations
frames = [create_input_arr_soc(x_train_fail, faulty=1), create_input_arr_soc(x_train_healthy, faulty=0)]
df_train_soc = pd.concat(frames, ignore_index=True)

# Create validation set only deviation
frames = [create_input_arr_soc(x_val_fail, faulty=1), create_input_arr_soc(x_val_healthy, faulty=0)]
df_val_soc = pd.concat(frames, ignore_index=True)


# ### Dataset/loaders

class MyDataset(Dataset):
    def __init__(self,df):
        x = df['Input'].values
        y = df['Label'].values
        x = [arr.astype('float32') for arr in x]
        y = [float(ele) for ele in y]
        self.x_train=torch.tensor(x)
        self.y_train=torch.tensor(y)

    def __len__(self):
        return len(self.y_train)
  
    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]


# +
val_set = MyDataset(df_val)
#test_set = MyDataset(df_test)
train_set = MyDataset(df_train)
#test_final_set = MyDataset(df_test_final)

batch_size = 10
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, drop_last=True)
#test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True,drop_last=True)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True,drop_last=True)
#test_final_loader = DataLoader(dataset=test_final_set, batch_size=batch_size, shuffle=True,drop_last=True)

# +
val_set_dev = MyDataset(df_val_dev)
#test_set = MyDataset(df_test)
train_set_dev = MyDataset(df_train_dev)
#test_final_set = MyDataset(df_test_final)

batch_size = 10
val_loader_dev = DataLoader(dataset=val_set_dev, batch_size=batch_size, shuffle=False, drop_last=True)
#test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True,drop_last=True)
train_loader_dev = DataLoader(dataset=train_set_dev, batch_size=batch_size, shuffle=True,drop_last=True)
#test_final_loader = DataLoader(dataset=test_final_set, batch_size=batch_size, shuffle=True,drop_last=True)

# +
val_set_soc = MyDataset(df_val_soc)
#test_set = MyDataset(df_test)
train_set_soc = MyDataset(df_train_soc)
#test_final_set = MyDataset(df_test_final)

batch_size = 10
val_loader_soc = DataLoader(dataset=val_set_soc, batch_size=batch_size, shuffle=False, drop_last=True)
#test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True,drop_last=True)
train_loader_soc = DataLoader(dataset=train_set_soc, batch_size=batch_size, shuffle=True,drop_last=True)
#test_final_loader = DataLoader(dataset=test_final_set, batch_size=batch_size, shuffle=True,drop_last=True)
# -

inputs, classes = next(iter(val_loader))
print(inputs.shape, classes.shape)

inputs, classes = next(iter(val_loader_dev))
print(inputs.shape, classes.shape)

inputs, classes = next(iter(val_loader_soc))
print(inputs.shape, classes.shape)


# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ### Normalize/Standardize data
# -

def get_mean_and_std(dataloader):
    inputs, _ = next(iter(dataloader))
    input_sum = np.zeros(inputs.shape[1])
    maxi = np.zeros(inputs.shape[1])
    mini = 1000*np.ones(inputs.shape[1])
    input_sq_sum = np.zeros(inputs.shape[1])
    num_batches = 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        maxi = np.maximum(maxi, data.max(axis=0)[0].detach().numpy())
        mini = np.minimum(mini, data.min(axis=0)[0].detach().numpy())
        input_sum += data.mean(axis=0).detach().numpy()
        input_sq_sum += (data**2).mean(axis=0).detach().numpy()
        num_batches += 1
    
    mean = input_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (input_sq_sum / num_batches - mean ** 2) ** 0.5

    return mean, std, maxi, mini


global mean_dev
global std_dev
global maxi
global mini 
mean_dev, std_dev, maxi, mini = get_mean_and_std(train_loader_dev)
def trfrm(sample):
    return (sample-mini)/(maxi-mini)


print('1:', mean_dev)
print('2:', std_dev)
print('3:', maxi)
print('4:', mini)

np.maximum([0,2,3],[1,2,0])

inputs, classes = next(iter(train_loader_dev))
print(inputs[0])

# ## Models

from torch import nn
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
import sklearn


# + [markdown] tags=[]
# ### Logistic Regression

# +
def convert_input_logreg(df):
    y = df['Label'].to_numpy()
    x = df['Input'].to_numpy() 
    x = np.hstack(x).reshape(x.shape[0],-1)
    x = [arr.astype('float32') for arr in x]
    y = [float(ele) for ele in y]
    return x, y

x_train, y_train = convert_input_logreg(df_train)
x_val, y_val = convert_input_logreg(df_val)
x_train_dev, y_train_dev = convert_input_logreg(df_train_dev)
x_val_dev, y_val_dev = convert_input_logreg(df_val_dev)
x_train_soc, y_train_soc = convert_input_logreg(df_train_soc)
x_val_soc, y_val_soc = convert_input_logreg(df_val_soc)
# -

for i in range(500):
    print(logReg.predict(x_val[i].reshape(1,-1)), logReg_dev.predict(x_val_dev[i].reshape(1,-1)))

logReg = LogisticRegression(max_iter = 10000)
logReg_dev = LogisticRegression(max_iter = 10000)
logReg_soc = LogisticRegression(max_iter = 10000)

logReg.fit(x_train, y_train)
logReg_dev.fit(x_train_dev, y_train_dev)
logReg_soc.fit(x_train_soc, y_train_soc)

pred = logReg.predict(x_val)
pred_dev = logReg_dev.predict(x_val_dev)
pred_soc = logReg_soc.predict(x_val_soc)

f1 = f1_score(pred, y_val)
tn, fp, fn, tp = confusion_matrix(pred, y_val).ravel()
print('True Negatives:', tn, 'True Positives:', tp, 'False negatives:', fn, 'False Positives:', fp)
print('F1 score:' ,f1)

f1_dev = f1_score(pred_dev, y_val_dev)
tn_dev, fp_dev, fn_dev, tp_dev = confusion_matrix(pred_dev, y_val_dev).ravel()
print('True Negatives:', tn_dev, 'True Positives:', tp_dev, 'False negatives:', fn_dev, 'False Positives:', fp_dev)
print('F1 score:', f1_dev)

f1_soc = f1_score(pred_soc, y_val_soc)
tn_soc, fp_soc, fn_soc, tp_soc = confusion_matrix(pred_soc, y_val_soc).ravel()
print('True Negatives:', tn_soc, 'True Positives:', tp_soc, 'False negatives:', fn_soc, 'False Positives:', fp_soc)
print('F1 score:', f1_soc)


# + [markdown] tags=[]
# ### ANN models
# -

# model definition
class MLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = nn.Linear(n_inputs, 10)
        nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = nn.ReLU()
        # second hidden layer
        self.hidden2 = nn.Linear(10, 8)
        nn.init.kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = nn.ReLU()
        # third hidden layer and output
        self.hidden3 = nn.Linear(8, 1)
        nn.init.xavier_uniform_(self.hidden3.weight)
        self.act3 = nn.Sigmoid()
 
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
         # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X


class ANNMultilayerperceptron(nn.Module):

    def __init__(self, input_size=1095,output_size=1, layers=[220,84]):  # 120, 84
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
        X = torch.sigmoid(self.fc3(X))

        #return F.log_softmax(X,dim=1) # PGA multiclass classification
        return X


class smallMLP(nn.Module):
    def __init__(self,input_shape):
        super(smallMLP,self).__init__()
        self.fc1 = nn.Linear(input_shape,8)
        self.fc2 = nn.Linear(8,4)
        self.fc3 = nn.Linear(4,1)
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class mediumMLP(nn.Module):
    def __init__(self,input_shape):
        super(mediumMLP,self).__init__()
        self.fc1 = nn.Linear(input_shape,108)
        self.fc2 = nn.Linear(108,36)
        self.fc3 = nn.Linear(36,1)
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# + [markdown] tags=[]
# ### Create model

# +
inputs, _ = next(iter(val_loader_soc))   
inp_shape = inputs.shape[1]

# model = ANNMultilayerperceptron(input_size=inp_shape)
# model = smallMLP(input_shape = inp_shape)
# model = MLP(inp_shape)
model = mediumMLP(input_shape = inp_shape)
# -

# Hyper parameters
learning_rate = 0.01
n_epochs = 100
# Optimizer, Loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.BCELoss()


# ### Training

# train the model
def train_model(train_dl, val_dl, model, n_epochs):
    model.train()
    # define the optimization
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    all_loss = list()
    # enumerate epochs
    for epoch in range(n_epochs):
        # enumerate mini batches
        train_loss_epoch = 0.0
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat.flatten(), targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            # Save loss values
            train_loss_epoch += loss.item()
        all_loss.append(train_loss_epoch)
        print(f"Train loss for epoch {epoch}:", train_loss_epoch/len(train_dl))
        
        val_loss_epoch = 0.0
        model.eval()     # Optional when not using Model Specific layer
        for data, labels in val_dl:
            # Forward Pass
            target = model(data)
            # Find the Loss
            loss = criterion(target.flatten(), labels)
            # Calculate Loss
            val_loss_epoch += loss.item()
        print(f"Val loss for epoch {epoch}:", val_loss_epoch/len(val_dl))



train_model(train_loader_soc, val_loader_soc, model, 30)


# ### Evaluation

# Evaluate the model
def evaluate_model(dl, model):
    model.eval()
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(dl):
        # Make prediction with the model
        with torch.no_grad():
            yhat = model(inputs)
        # Retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # Round to class values
        yhat = yhat.round()
        # Store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    # Compute evaluation metrics
    conf_mat = confusion_matrix(actuals, predictions)
    prec = precision_score(actuals, predictions)
    rec = recall_score(actuals, predictions)
    f1 = f1_score(actuals, predictions)
    return f1, conf_mat, prec, rec


f1, conf_mat, prec, rec = evaluate_model(val_loader_soc, model)
tn, fp, fn, tp = conf_mat.ravel()
print('True Negatives:', tn, 'True Positives:', tp, 'False negatives:', fn, 'False Positives:', fp)
print('F1 score:', f1)

f1, conf_mat, prec, rec = evaluate_model(train_loader_soc, model)
tn, fp, fn, tp = conf_mat.ravel()
print('True Negatives:', tn, 'True Positives:', tp, 'False negatives:', fn, 'False Positives:', fp)
print('F1 score:', f1)


