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

#-----------------------
# function
def readDataSet(dir_path):
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

    return cntClient, listClientTrain, dfServerVal, dfServerTest


def plotInfo(plot_dir_path, bPlotClientTrainInfo, bPlotTestValInfo, listClientTrain, dfServerVal, dfServerTest):

    listClientTrainInfo = [df[['SensitiveAttr', 'Label']] for df in listClientTrain]
    dfServerTestInfo = dfServerTest[['SensitiveAttr', 'Label']]
    dfServerValInfo = dfServerVal[['SensitiveAttr', 'Label']]

    if bPlotClientTrainInfo:
        client.PlotClientDataDist(listClientTrainInfo, plot_dir_path, todayTime, "training")

    if bPlotTestValInfo:
        client.PlotServerDataDist(dfServerTestInfo, plot_dir_path, todayTime, "testing")
        client.PlotServerDataDist(dfServerValInfo, plot_dir_path, todayTime, "validation")

####################(Initialized program setting)########################
bRunSplitDataSet = True
bPlotClientTrainInfo = True
bPlotTestValInfo = True
bOutputExcel = True
today_path = helper.CreateTodayFolder()
todayTime = helper.GetTodayTime()

rTrain = 0.6
rVal = 0.2
rTest = 0.2
alpha = 2 # [0.5, 2]
num_clients = 10

####################(Initialized Alg setting)########################
strFedAlg = "FairFateVC" # [FairFate, FairFateVC]
intRun = 10
intGlobalIteration = 50
rSubsetClient = 0.3
METRICS_values = ['SP', 'EO', 'EQO'] # ['EO'] # ['SP', 'EO', 'EQO']
beta0_values = [0.9] # initial Momentum parameter: [0.8, 0.9, 0.99]
rho_values = [0.04] # growth rate: [0.04, 0.05]
lr_values = [0.01] # learning rate: [0.01, 0.02]
lambda0_values = [0.5] # initial fairness amount: [0.1, 0.5]
MAX_values = [0.9] # maximum fairness amount: [0.8, 0.9, 1.0]

# strFedAlg = "FairFate" # [FedAvg, FairFate, FairFateVC]
# intRun = 10
# intGlobalIteration = 100
# rSubsetClient = 0.3
# METRICS_values = ['SP', 'EO', 'EQO']
# beta0_values = [0.8, 0.9, 0.99]
# rho_values = [0.04, 0.05]
# lr_values = [0.01, 0.02]
# lambda0_values = [0.1, 0.5]
# MAX_values = [0.8, 0.9, 1.0]

################################################

dfFairResult = pd.DataFrame()
dfAccResult = pd.DataFrame()

for strFairMatric in METRICS_values:
    fairnessCollection = np.zeros((1, intGlobalIteration))
    accCollection = np.zeros((1, intGlobalIteration))

    n = 0
    for beta0 in beta0_values:
        for rho in rho_values:
            for lr in lr_values:
                for lambda0 in lambda0_values:
                    for max in MAX_values:   
                        for run in range(intRun): 

                            if bRunSplitDataSet:
                                # re-create data set
                                dir_path = dataset.split_existed_to_train_val_test(today_path, strFairMatric, run, rTrain, rVal, rTest, alpha, num_clients)
                            else:
                                dir_path = 'today_20230530\\dataset_SP_run_0' # 'today_20230530\\dataset_SP_run_0'

                            result_dir_path = helper.CreateResultFolder(dir_path, todayTime, strFedAlg)

                            f = open(result_dir_path + f'\\output_Main_{todayTime}.out', 'w')
                            original = sys.stdout
                            sys.stdout = Tee(sys.stdout, f)

                            # read dataset
                            cntClient, listClientTrain, dfServerVal, dfServerTest = readDataSet(dir_path)

                            # plot
                            if bPlotClientTrainInfo or bPlotTestValInfo:
                                plot_dir_path = helper.CreatePlotFolder(dir_path, todayTime, strFedAlg)
                                plotInfo(plot_dir_path, bPlotClientTrainInfo, bPlotTestValInfo, listClientTrain, dfServerVal, dfServerTest)

                            # run alg.
                            n += 1         
                            fairnessResult, accResult = FLAlgMain.main(strFedAlg, beta0, rho, lr, lambda0, max, strFairMatric,
                                        listClientTrain, dfServerVal, dfServerTest,
                                        run, cntClient, intGlobalIteration, rSubsetClient,
                                        bOutputExcel, result_dir_path)
                            
                            fairnessCollection = fairnessCollection + fairnessResult
                            accCollection = accCollection + accResult

                            sys.stdout = original
                            f.close()

    # compute avg for each strFairMatric
    fairnessCollection /= n
    accCollection /= n
    dfFairResult[f'fair_{strFairMatric}'] = pd.Series(fairnessCollection.flatten())
    dfAccResult[f'acc_{strFairMatric}'] = pd.Series(accCollection.flatten())

dfFairResult.to_csv(f'{today_path}\\00_dfFairResult_{strFedAlg}_{todayTime}.csv', index=False)
dfAccResult.to_csv(f'{today_path}\\00_dfAccResult_{strFedAlg}_{todayTime}.csv', index=False)





