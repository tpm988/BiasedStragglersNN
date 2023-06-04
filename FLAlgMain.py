import os
import os.path
import pandas as pd
import numpy as np
import warnings
# from datetime import datetime
#-----------------------
import server as s
import model as m
# import client
#-----------------------
from sklearn.exceptions import ConvergenceWarning

# Ignore ConvergenceWarning and FutureWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.system('cls||clear')

################################################

def main(strFedAlg, beta0, rho, lr, lambda0, max, strFairMatric,
        listClientTrain, dfServerVal, dfServerTest,
        run, cntClient, intGlobalIteration, rSubsetClient,
        bOutputExcel, dir_path):

    # Momentum update
    v = 0

    print ('----------------Execute (FLAlgMain)----------------')

####################(Training model start)########################

    cntFeature = listClientTrain[0].shape[1] - 1

    # initialize a dataframe to store the client model results:
    colClientAccFair = ['iter', 'client', 'numData', 'Acc', 'fairType', 'fairValue', 'Privileged', 'higher']
    dfClientAccFair = pd.DataFrame(columns = colClientAccFair)

    # global model coefficient and client data accuracy record
    colServerDataAccFair = ['iter', 'dataType', 'Acc', 'fairType', 'fairValue', 'Privileged'] # dataType = ['test', 'val']
    dfServerDataAccFair = pd.DataFrame(columns = colServerDataAccFair)

    print (f'----------------(start running algorithm)----------------')

    # (t = 0): initial weights and biases
    model, weights_h1, bias_h1, weights_out, bias_out = m.CreatModelNNInit(cntFeature, lr, run+1)

    dfServerDataAccFair = s.calFairness(0, False, "val", strFairMatric, model, dfServerVal, dfServerDataAccFair, F_Global=0, strPrivileged='Y', cIdx=0, numData=0)
    F_Global = dfServerDataAccFair[(dfServerDataAccFair["iter"] == 0) & (dfServerDataAccFair["dataType"] == "val")].fairValue.values[0]
    strPrivileged = dfServerDataAccFair[(dfServerDataAccFair["iter"] == 0) & (dfServerDataAccFair["dataType"] == "val")].Privileged.values[0]

    for t in range(1, intGlobalIteration + 1 ):

        print(f"------------ run = {run+1}; t = {t}; F_Global = {F_Global}, Privileged = {strPrivileged} ------------")

        # copy model for client used
        model_global = m.CloneModelNN(model, lr)

        # select subset of client
        subsetOfClient = np.sort(np.random.choice(range(1, cntClient+1), size=round(cntClient*rSubsetClient), replace=False))
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
            
            # copy model for client used
            model_client = m.CloneModelNN(model_global, lr)

            # neural network
            model_client, client_HidCoef, client_HidBias, client_OutCoef, client_OutBias = m.train_NN_SGD(t, cIdx, dataset, model_client, lr,
                                                                        client_HidCoef, client_HidBias,
                                                                        client_OutCoef, client_OutBias)

            # start record client model evaluation from iteration 1
            dfClientAccFair = s.calFairness(t, True, "val", strFairMatric, model_client, dfServerVal, dfClientAccFair, F_Global, strPrivileged, cIdx, numData)

        # select subset of client with higher fairness
        dfClientAccFairTemp  = dfClientAccFair[dfClientAccFair["iter"] == t].reset_index(drop=True)
        cIdxFair = dfClientAccFairTemp[dfClientAccFairTemp["higher"] == "Y"].client.to_numpy()

        # calculate alphaF
        alphaF_weights_h1, alphaF_bias_h1, alphaF_weights_out, alphaF_bias_out = s.cal_alphaF(dfClientAccFairTemp[dfClientAccFairTemp["client"].isin(cIdxFair)],
                            client_HidCoef, client_HidBias, client_OutCoef, client_OutBias,
                            weights_h1, bias_h1, weights_out, bias_out
                            )

        # calculate alphaN
        alphaN_weights_h1, alphaN_bias_h1, alphaN_weights_out, alphaN_bias_out = s.cal_alphaN(dfClientAccFairTemp,
                            client_HidCoef, client_HidBias, client_OutCoef, client_OutBias,
                            weights_h1, bias_h1, weights_out, bias_out
                            )
        beta = beta0 * (1-t/intGlobalIteration) / ((1-beta0) + (beta0 * (1-t/intGlobalIteration)))

        v_weights_h1 = beta*v + (1-beta)*alphaF_weights_h1
        v_bias_h1 = beta*v + (1-beta)*alphaF_bias_h1
        v_weights_out = beta*v + (1-beta)*alphaF_weights_out
        v_bias_out = beta*v + (1-beta)*alphaF_bias_out

        lambda_t = lambda0*(1 + rho)**t

        if lambda_t >= max:
            lambda_t = max

        # update theta
        weights_h1 = weights_h1 + lambda_t*v_weights_h1 + (1-lambda_t)*alphaN_weights_h1
        bias_h1 = bias_h1 + lambda_t*v_bias_h1 + (1-lambda_t)*alphaN_bias_h1
        weights_out = weights_out + lambda_t*v_weights_out + (1-lambda_t)*alphaN_weights_out
        bias_out = bias_out + lambda_t*v_bias_out + (1-lambda_t)*alphaN_bias_out

        # calculate fairness on validation/testing set
        model = m.CreatModelNN(cntFeature, lr, weights_h1, bias_h1, weights_out, bias_out)
        dfServerDataAccFair = s.calFairness(t, False, "val", strFairMatric, model, dfServerVal, dfServerDataAccFair, F_Global=0, strPrivileged='Y', cIdx=0, numData=0)
        dfServerDataAccFair = s.calFairness(t, False, "test", strFairMatric, model, dfServerTest, dfServerDataAccFair, F_Global=0, strPrivileged='Y', cIdx=0, numData=0)

        F_Global = dfServerDataAccFair[(dfServerDataAccFair["iter"] == t) & (dfServerDataAccFair["dataType"] == "val")].fairValue.values[0]
        strPrivileged = dfServerDataAccFair[(dfServerDataAccFair["iter"] == t) & (dfServerDataAccFair["dataType"] == "val")].Privileged.values[0]

        if (strFedAlg == 'FairFateVC'):
            cIdxUnfair = np.setdiff1d(subsetOfClient, cIdxFair)
            if len(cIdxUnfair) > 0:
                # split virtual client dataset
                listClientTrainUnfair = [listClientTrain[i-1] for i in cIdxUnfair]
                for i in sorted(cIdxUnfair, reverse=True):
                    del listClientTrain[i-1]
                listClientTrainUnfair = s.FairFateVC(listClientTrainUnfair)
                # update listClientTrain & cntClient
                listClientTrain += listClientTrainUnfair
                cntClient = len(listClientTrain)

            # if round(cntClient*rSubsetClient) == 6:
            #     plotMore = False
            #     if plotMore == True:
            #         plotMore = False
            #         listClientTrainInfo = [df[['SensitiveAttr', 'Label']] for df in listClientTrain]
            #         todayTime = datetime.now().strftime("%Y%m%d %H.%M.%S")
            #         client.PlotClientDataDist(listClientTrainInfo, dir_path, todayTime, "training")

            # if round(cntClient*rSubsetClient) = 7:
            #     plotMore = False
            #     if plotMore == True:
            #         plotMore = False
            #         listClientTrainInfo = [df[['SensitiveAttr', 'Label']] for df in listClientTrain]
            #         todayTime = datetime.now().strftime("%Y%m%d %H.%M.%S")
            #         client.PlotClientDataDist(listClientTrainInfo, dir_path, todayTime, "training")      
            

    if bOutputExcel:
        dfClientAccFair.to_csv(f'{dir_path}\\dfClientAccFair_{strFedAlg}_f_{strFairMatric}_run_{run+1}_beta_{beta0}_rho_{rho}_lr_{lr}_lb0_{lambda0}_max_{max}.csv', index=False)
        dfServerDataAccFair.to_csv(f'{dir_path}\\dfServerDataAccFair_{strFedAlg}_f_{strFairMatric}_run_{run+1}_beta_{beta0}_rho_{rho}_lr_{lr}_lb0_{lambda0}_max_{max}.csv', index=False)
        dfServerDataAccFair[dfServerDataAccFair["dataType"] == "test"].iloc[:,[0,2,4]].to_csv(f'{dir_path}\\dfServerDataAccFair_test_{strFedAlg}_f_{strFairMatric}_run_{run+1}_beta_{beta0}_rho_{rho}_lr_{lr}_lb0_{lambda0}_max_{max}.csv', index=False)

    fairnessResult = np.array(dfServerDataAccFair[dfServerDataAccFair["dataType"] == "test"].fairValue)
    accResult = np.array(dfServerDataAccFair[dfServerDataAccFair["dataType"] == "test"].Acc)

    ################################################

    print ('----------------Done (FLAlgMain)----------------')

    return fairnessResult, accResult

