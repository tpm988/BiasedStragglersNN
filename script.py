import os
import os.path
import sys
from datetime import date
import fnmatch
import pandas as pd
import numpy as np
#-----------------------
import client
import helper
import dataset
import FLAlgMain

#-----------------------
# Define the Tee class
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for file in self.files:
            file.write(obj)
            file.flush() # If you want the output to be visible immediately
    def flush(self):
        for file in self.files:
            file.flush()

####################(Initialized program setting)########################
bRunSplitDataSet = False
bPlotClientTrainInfo = False
bPlotTestValInfo = False
bOutputExcel = True
todayTime = helper.GetTodayTime()

if bRunSplitDataSet:
    cntClient = 10
    alpha = 1
    dataset.split_existed_to_train_val_test(rTrain=0.6, rVal=0.2, rTest=0.2, alpha=alpha, num_clients=cntClient)
    today = date.today().strftime("%Y%m%d")
    dir_path = f'dataset_{today}'
else:
    today = "20230526"
    dir_path = f'dataset_{today}'

####################(Initialized Alg setting)########################
strFedAlg = "FairFate" # [FedAvg, FairFate, FairFateVC]
intRun = 3
intGlobalIteration = 5
rSubsetClient = 0.5
METRICS_values = ['SP']
beta0_values = [0.8] # initial Momentum parameter: [0.8, 0.9, 0.99]
rho_values = [0.04] # growth rate: [0.04, 0.05]
lr_values = [0.01] # learning rate: [0.01, 0.02]
lambda0_values = [0.5] # initial fairness amount: [0.1, 0.5]
MAX_values = [0.8] # maximum fairness amount: [0.8, 0.9, 1.0]

# strFedAlg = "FairFate" # [FedAvg, FairFate, FairFateVC]
# intRun = 10
# intGlobalIteration = 100
# rSubsetClient = 0.5
# METRICS_values = ['SP', 'EO', 'EQO']
# beta0_values = [0.8, 0.9, 0.99]
# rho_values = [0.04, 0.05]
# lr_values = [0.01, 0.02]
# lambda0_values = [0.1, 0.5]
# MAX_values = [0.8, 0.9, 1.0]

if bPlotClientTrainInfo or bPlotTestValInfo:
    plot_dir_path = helper.CreatePlotFolder(dir_path, todayTime, strFedAlg)

if bOutputExcel:
    result_dir_path = helper.CreateResultFolder(dir_path, todayTime, strFedAlg)

################################################
# Open the file you want to write to
f = open(result_dir_path + f'\\output_Main_{todayTime}.out', 'w')

# Save the original stdout object for later
original = sys.stdout

# Replace stdout with a Tee, so output goes to stdout and the file
sys.stdout = Tee(sys.stdout, f)

################################################
cntClient = len(fnmatch.filter(os.listdir(dir_path), 'training_data_client_*.csv'))
print(f'Alg.: {strFedAlg}; Client dataset: {dir_path}; client Count: {cntClient}')

# read clients training dataset
print (f'----------------(start loading {cntClient} client datasets from {dir_path})----------------')
file_name_train = dir_path + '\\training_data_client_{}.csv'
file_name_test = dir_path + '\\testing_data.csv'
file_name_val = dir_path + '\\validation_data.csv'
listClientTrain = []
for i in range(1, cntClient + 1):
    listClientTrain.append(pd.read_csv(file_name_train.format(i)))

dfServerTest = pd.read_csv(file_name_test)
dfServerVal = pd.read_csv(file_name_val)
print (f'----------------(successfully loading {cntClient} client datasets and server testing data from {dir_path})----------------')

listClientTrainInfo = [df[['SensitiveAttr', 'Label']] for df in listClientTrain]
dfServerTestInfo = dfServerTest[['SensitiveAttr', 'Label']]
dfServerValInfo = dfServerVal[['SensitiveAttr', 'Label']]

####################(Plot training data)########################

if bPlotClientTrainInfo:
    client.PlotClientDataDist(listClientTrainInfo, plot_dir_path, todayTime, "training")

####################(Plot testing data)########################

if bPlotTestValInfo:
    client.PlotServerDataDist(dfServerTestInfo, plot_dir_path, todayTime, "testing")
    client.PlotServerDataDist(dfServerValInfo, plot_dir_path, todayTime, "validation")

################################################
dfResult = pd.DataFrame()

for strFairMatric in METRICS_values:
    fairnessCollection = np.zeros((1, intGlobalIteration))
    accCollection = np.zeros((1, intGlobalIteration))
    n = 0
    for run in range(intRun):   
        for beta0 in beta0_values:
            for rho in rho_values:
                for lr in lr_values:
                    for lambda0 in lambda0_values:
                        for max in MAX_values:      
                            n += 1              
                            fairnessResult, accResult = FLAlgMain.main(strFedAlg, beta0, rho, lr, lambda0, max, strFairMatric,
                                        listClientTrain, dfServerVal, dfServerTest,
                                        run, cntClient, intGlobalIteration, rSubsetClient,
                                        bOutputExcel, todayTime, result_dir_path)
                            
                            fairnessCollection = fairnessCollection + fairnessResult
                            accCollection = accCollection + accResult

    # compute avg for each strFairMatric
    fairnessCollection /= n
    accCollection /= n
    dfResult[f'fair_{strFairMatric}'] = pd.Series(fairnessCollection.flatten())
    dfResult[f'acc_{strFairMatric}'] = pd.Series(accCollection.flatten())

dfResult.to_csv(f'{result_dir_path}\\00_dfResult_{today}_{strFedAlg}.csv', index=False)

################################################
# At the end of the script, restore stdout and close the file
sys.stdout = original
f.close()