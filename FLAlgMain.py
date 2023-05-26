import os
import os.path
import sys
import pandas as pd
import numpy as np
import fnmatch
import warnings
from datetime import date
from pandas import ExcelWriter
#-----------------------
import dataset
import client
import server as s
import model as m
import helper
#-----------------------
from sklearn.exceptions import ConvergenceWarning

# Ignore ConvergenceWarning and FutureWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.system('cls||clear')

####################(Initialized program setting)########################
bRunSplitDataSet = False
bPlotClientTrainInfo = True
bPlotTestValInfo = True
bPlotClientAccEOD = False
bOutputExcel = True
# bRunTesting = True
todayTime = helper.GetTodayTime()

####################(Initialized Alg setting)########################
strFedAlg = "FairFate" # [FedAvg, FairFate, FairFateVC]
strFairMatric = "EQO" #['SP', 'EO', 'EQO']
intGlobalIteration = 100
rSubsetClient = 0.5
beta0 = 0.8 # initial Momentum parameter: [0.8, 0.9, 0.99]
rho = 0.04 # growth rate: [0.04, 0.05]
lambda0 = 0.5 # initial fairness amount: [0.1, 0.5]
MAX = 0.8 # maximum fairness amount: [0.8, 0.9, 1.0]
v = 0 # Momentum update

if bRunSplitDataSet:
    cntClient = 10
    alpha = 1
    dfTrain, dfVal, dfTest, client_datasets = dataset.split_train_val_test(rTrain=0.6, rVal=0.2, rTest=0.2, alpha=alpha, num_clients=cntClient)
    today = date.today().strftime("%Y%m%d")
    dir_path = f'dataset_{today}'
else:
    today = "20230525"
    dir_path = f'dataset_{today}'

if bPlotClientTrainInfo or bPlotClientAccEOD or bPlotTestValInfo:
    plot_dir_path = helper.CreatePlotFolder(dir_path, todayTime, strFedAlg)

################################################
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

# Open the file you want to write to
f = open(dir_path + '\\output_AlgMain.out', 'w')

# Save the original stdout object for later
original = sys.stdout

# Replace stdout with a Tee, so output goes to stdout and the file
sys.stdout = Tee(sys.stdout, f)

print ('----------------Execute (FLAlgMain)----------------')

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

####################(Training model start)########################

# initialize a dataframe to store the client model results:
cntFeature = listClientTrain[0].shape[1] - 1
colClientCoef = ['iter', 'client', 'numData', 'train_acc', 'intercept'] + ['coef_x_%d'%i for i in range(1, cntFeature + 1)]
aryGlobalCoef = []
fGlobalInterception = None
dfClientCoef = pd.DataFrame(columns = colClientCoef)

# global model coefficient and client data accuracy record
colGlobalCoef = ['iter', 'intercept'] + ['coef_x_%d'%i for i in range(1, cntFeature + 1)]
dfGlobalCoef = pd.DataFrame(columns = colGlobalCoef) 
colClientAccFair = ['iter', 'client', 'Acc', 'fairType', 'fairValue', 'higher']
dfClientAccFair = pd.DataFrame(columns = colClientAccFair)
colServerDataAccFair = ['iter', 'dataType', 'Acc', 'fairType', 'fairValue'] # dataType = ['test', 'val']
dfServerDataAccFair = pd.DataFrame(columns = colServerDataAccFair)

print (f'----------------(start running algorithm)----------------')

for t in range(0, intGlobalIteration + 1 ):

    print(f"------------ t = {t} ------------")

    # select subset of client
    subsetOfClient = np.sort(np.random.choice(range(1, cntClient+1), size=int(cntClient*rSubsetClient), replace=False))

    # train client model
    for k in range(len(subsetOfClient)):
        
        # client index
        cIdx = subsetOfClient[k]
        dataset = listClientTrain[cIdx-1]

        # neural network
        resultCoef, aryClientCoef, fClientInterception = m.train_Logistic_Regression(t, cIdx, dataset, aryGlobalCoef, fGlobalInterception)
        dfClientCoef.loc[len(dfClientCoef.index),0:] = resultCoef

        if t > 0:
            dfClientAccFair = s.calFairness(t, True, cIdx, strFairMatric, aryClientCoef, fClientInterception, dfServerVal, dfClientAccFair, F_Global)

    if t == 0: 
        # initial global model coef: FedAvg
        aryGlobalCoef, fGlobalInterception = s.FedAvg(t, cntFeature, dfClientCoef[dfClientCoef['iter'] == t])
        resultGlobalCoef = [t]
        resultGlobalCoef.extend([fGlobalInterception])
        resultGlobalCoef.extend(aryGlobalCoef)
        dfGlobalCoef.loc[len(dfGlobalCoef.index),0:] = resultGlobalCoef

    else:
        # select subset of client with higher fairness 
        cIdxFair = dfClientAccFair[(dfClientAccFair["iter"] == t) & (dfClientAccFair["higher"] == "Y")].client.to_numpy()
        F_total = dfClientAccFair[(dfClientAccFair["iter"] == t) & (dfClientAccFair["client"].isin(cIdxFair))].fairValue.sum()

        if len(cIdxFair) > 0:
            alphaF = s.cal_alphaF(cntFeature,
                                dfClientAccFair[(dfClientAccFair["iter"] == t) & (dfClientAccFair["client"].isin(cIdxFair))],
                                dfClientCoef[(dfClientCoef["iter"] == t) & (dfClientCoef["client"].isin(cIdxFair))],
                                dfGlobalCoef[dfGlobalCoef["iter"] == (t-1)]
                                )
        else:
            alphaF = np.zeros(1+cntFeature)

        alphaN = s.cal_alphaN(cntFeature, 
                            dfClientCoef[dfClientCoef["iter"] == t],
                            dfGlobalCoef[dfGlobalCoef["iter"] == (t-1)]
                            )
        beta = beta0 * (1-t/intGlobalIteration) / ((1-beta0) + (beta0 * (1-t/intGlobalIteration)))
        v = beta*v + (1-beta)*alphaF
        lambda_t = lambda0*(1 + rho)**t
        if lambda_t >= MAX:
            lambda_t = MAX

        theta_t = dfGlobalCoef[dfGlobalCoef["iter"] == (t-1)].iloc[:,1:].values.flatten()
        theta_tp1 = theta_t + lambda_t*v + (1-lambda_t)*alphaN

        fGlobalInterception = theta_tp1[0]
        aryGlobalCoef = list(theta_tp1[1:])

        resultGlobalCoef = [t]
        resultGlobalCoef.extend(list(theta_tp1))
        dfGlobalCoef.loc[len(dfGlobalCoef.index),0:] = resultGlobalCoef

    # calculate fairness on validation/testing set
    dfServerDataAccFair = s.calFairness(t, False, 'val', strFairMatric, aryGlobalCoef, fGlobalInterception, dfServerVal, dfServerDataAccFair)
    dfServerDataAccFair = s.calFairness(t, False, 'test', strFairMatric, aryGlobalCoef, fGlobalInterception, dfServerTest, dfServerDataAccFair)

    F_Global = dfServerDataAccFair[(dfServerDataAccFair["iter"] == t) & (dfServerDataAccFair["dataType"] == "val")].fairValue.values[0]



# dfServerDataAccFair.to_csv(f'{dir_path}\\dfServerDataAccFair_{today}_{strFedAlg}_{strFairMatric}.csv', index=False)

dfServerDataAccFair[dfServerDataAccFair["dataType"] == "test"].iloc[:,[0,2,4]].to_csv(f'{dir_path}\\dfServerDataAccFair_{today}_{strFedAlg}_{strFairMatric}.csv', index=False)

################################################

print ('----------------Done (FLAlgMain)----------------')

# At the end of the script, restore stdout and close the file
sys.stdout = original
f.close()