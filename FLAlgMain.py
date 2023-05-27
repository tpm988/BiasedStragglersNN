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
bPlotClientTrainInfo = False
bPlotTestValInfo = False
bOutputExcel = True
todayTime = helper.GetTodayTime()

####################(Initialized Alg setting)########################
strFedAlg = "FairFate" # [FedAvg, FairFate, FairFateVC]
strFairMatric = "EO" #['SP', 'EO', 'EQO']
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
    today = "20230526"
    dir_path = f'dataset_{today}'

if bPlotClientTrainInfo or bPlotTestValInfo:
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
f = open(dir_path + f'\\output_AlgMain_{todayTime}.out', 'w')

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

cntFeature = listClientTrain[0].shape[1] - 1

# initialize a dataframe to store the client model results:
colClientAccFair = ['iter', 'client', 'numData', 'Acc', 'fairType', 'fairValue', 'higher']
dfClientAccFair = pd.DataFrame(columns = colClientAccFair)

# global model coefficient and client data accuracy record
colServerDataAccFair = ['iter', 'dataType', 'Acc', 'fairType', 'fairValue'] # dataType = ['test', 'val']
dfServerDataAccFair = pd.DataFrame(columns = colServerDataAccFair)

print (f'----------------(start running algorithm)----------------')

# (t = 0): initial weights and biases
weights_h1 = np.random.rand(cntFeature, 10) # for the first dense layer
bias_h1 = np.random.rand(10)
weights_out = np.random.rand(10, 1) # for the second dense layer
bias_out = np.random.rand(1)
model = m.CreatModelNNInit(cntFeature, weights_h1, bias_h1, weights_out, bias_out)
dfServerDataAccFair = s.calFairness(0, False, "val", strFairMatric, model, dfServerVal, dfServerDataAccFair, F_Global=0, cIdx=0, numData=0)
F_Global = dfServerDataAccFair[(dfServerDataAccFair["iter"] == 0) & (dfServerDataAccFair["dataType"] == "val")].fairValue.values[0]

for t in range(1, intGlobalIteration + 1 ):

    print(f"------------ t = {t} ------------")

    # copy model for client used
    model_copy = m.CloneModelNN(model)

    # select subset of client
    subsetOfClient = np.sort(np.random.choice(range(1, cntClient+1), size=int(cntClient*rSubsetClient), replace=False))
    client_HidCoef = []
    client_HidBias = []
    client_OutCoef = []
    client_OutBias = []

    # train client model
    for k in range(len(subsetOfClient)):
        
        # client index
        cIdx = subsetOfClient[k]
        dataset = listClientTrain[cIdx-1]
        numData = dataset.shape[0]

        # neural network
        model_client, client_HidCoef, client_HidBias, client_OutCoef, client_OutBias = m.train_NN_SGD(t, cIdx, cntFeature, dataset, dfClientAccFair, model_copy,
                                                                       client_HidCoef, client_HidBias,
                                                                       client_OutCoef, client_OutBias)

        # start record client model evaluation from iteration 1
        dfClientAccFair = s.calFairness(t, True, "val", strFairMatric, model_client, dfServerVal, dfClientAccFair, F_Global, cIdx, numData)

    # select subset of client with higher fairness 
    cIdxFair = dfClientAccFair[(dfClientAccFair["iter"] == t) & (dfClientAccFair["higher"] == "Y")].client.to_numpy()
    F_total = dfClientAccFair[(dfClientAccFair["iter"] == t) & (dfClientAccFair["client"].isin(cIdxFair))].fairValue.sum()

    # calculate alphaF
    alphaF_weights_h1, alphaF_bias_h1, alphaF_weights_out, alphaF_bias_out = s.cal_alphaF(dfClientAccFair[(dfClientAccFair["iter"] == t) & (dfClientAccFair["client"].isin(cIdxFair))],
                        client_HidCoef, client_HidBias, client_OutCoef, client_OutBias,
                        weights_h1, bias_h1, weights_out, bias_out
                        )

    # calculate alphaN
    alphaN_weights_h1, alphaN_bias_h1, alphaN_weights_out, alphaN_bias_out = s.cal_alphaN(dfClientAccFair[dfClientAccFair["iter"] == t],
                        client_HidCoef, client_HidBias, client_OutCoef, client_OutBias,
                        weights_h1, bias_h1, weights_out, bias_out
                        )
    beta = beta0 * (1-t/intGlobalIteration) / ((1-beta0) + (beta0 * (1-t/intGlobalIteration)))

    v_weights_h1 = beta*v + (1-beta)*alphaF_weights_h1
    v_bias_h1 = beta*v + (1-beta)*alphaF_bias_h1
    v_weights_out = beta*v + (1-beta)*alphaF_weights_out
    v_bias_out = beta*v + (1-beta)*alphaF_bias_out

    lambda_t = lambda0*(1 + rho)**t

    if lambda_t >= MAX:
        lambda_t = MAX

    # update theta
    weights_h1 = weights_h1 + lambda_t*v_weights_h1 + (1-lambda_t)*alphaN_weights_h1
    bias_h1 = bias_h1 + lambda_t*v_bias_h1 + (1-lambda_t)*alphaN_bias_h1
    weights_out = weights_out + lambda_t*v_weights_out + (1-lambda_t)*alphaN_weights_out
    bias_out = bias_out + lambda_t*v_bias_out + (1-lambda_t)*alphaN_bias_out

    # calculate fairness on validation/testing set
    model = m.CreatModelNNInit(cntFeature, weights_h1, bias_h1, weights_out, bias_out)
    dfServerDataAccFair = s.calFairness(t, False, "val", strFairMatric, model, dfServerVal, dfServerDataAccFair, F_Global, cIdx=0, numData=0)
    dfServerDataAccFair = s.calFairness(t, False, "test", strFairMatric, model, dfServerTest, dfServerDataAccFair, F_Global, cIdx=0, numData=0)

    F_Global = dfServerDataAccFair[(dfServerDataAccFair["iter"] == 0) & (dfServerDataAccFair["dataType"] == "val")].fairValue.values[0]


if bOutputExcel:
    dfClientAccFair.to_csv(f'{dir_path}\\dfClientAccFair_{today}_{strFedAlg}_{strFairMatric}.csv', index=False)
    dfServerDataAccFair.to_csv(f'{dir_path}\\dfServerDataAccFair_{today}_{strFedAlg}_{strFairMatric}.csv', index=False)
    dfServerDataAccFair[dfServerDataAccFair["dataType"] == "test"].iloc[:,[0,2,4]].to_csv(f'{dir_path}\\dfServerDataAccFair_{today}_{strFedAlg}_{strFairMatric}.csv', index=False)

################################################

print ('----------------Done (FLAlgMain)----------------')

# At the end of the script, restore stdout and close the file
sys.stdout = original
f.close()