import model as m
import client as c
#-----------------------
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

######(Aggregate global model coefficient)######
def FairFateVC(listClientTrainUnfair):

    cntClient = len(listClientTrainUnfair)
    listClientTrain = []

    for i in range(cntClient):
        df = listClientTrainUnfair[i]
        if len(df) > 200:
            dfSplit = ClientTrainSplit(df)
            listClientTrain += dfSplit
        else:
            listClientTrain += df

    return listClientTrain

def ClientTrainSplit(df):
    df_00 = df[(df['SensitiveAttr'] == 0) & (df['Label'] == 0)]
    df_01 = df[(df['SensitiveAttr'] == 0) & (df['Label'] == 1)]
    df_10 = df[(df['SensitiveAttr'] == 1) & (df['Label'] == 0)]
    df_11 = df[(df['SensitiveAttr'] == 1) & (df['Label'] == 1)]

    min_len = min(len(df_00), len(df_01), len(df_10), len(df_11))

    # Check if can split: want len(df) >= 100
    if min_len < 25:
        return [df]
    
    # Calculate select_len: want len(df) <= len(df)/2
    select_len = min(min_len, int(len(df) / 8))

    df1 = pd.concat([df_00[:select_len], df_01[:select_len], df_10[:select_len], df_11[:select_len]])
    df2 = df.drop(df1.index)

    df1 = shuffle(df1)
    df2 = shuffle(df2)

    return [df1, df2]


def ModelEval(t, model, dfServerData):

    X_, y_ = c.SplitClientTrainValDataSet(dfServerData)
    predictions = model.predict(X_, verbose=0)
    predicted_labels = (predictions > 0.5).astype(int)
    loss, Acc = model.evaluate(X_, y_, verbose=0)

    # return sensitive attribute and prediction result
    df = dfServerData[['SensitiveAttr', 'Label']]
    df.insert(0, "iter", t, True)
    cntdfCol = len(df.columns)
    df.insert(cntdfCol, "Prediction", predicted_labels, True) # add Prediction to last column

    return Acc, df

def calFairness(t, bIsClient, DataType, strFairMatric, model, dfServerData, dfRecordAccFair, F_Global = 0, strPrivileged = 'Y', cIdx=0, numData=0):

    dfRecordAccFair.loc[len(dfRecordAccFair.index), 'iter'] = t
    idx = dfRecordAccFair[dfRecordAccFair['iter'] == t].index[-1]

    # create model and predict
    Acc, df = ModelEval(t, model, dfServerData)

    # evaluate fairness
    if strFairMatric == "SP":
        fairness, evaPrivileged = m.calSP(df, bIsClient, strPrivileged)
    elif strFairMatric == "EO":
        fairness, evaPrivileged = m.calEO(df, bIsClient, strPrivileged)
    elif strFairMatric == "EQO":
        fairness, evaPrivileged = m.calEQO(df, bIsClient, strPrivileged)

    if bIsClient:
        print(f'client {cIdx} accuracy on {DataType}: %.6f' % (Acc))
        print(f'client {cIdx} fairness({strFairMatric}, {strPrivileged}) on {DataType}: %.6f' % (fairness))
        dfRecordAccFair.at[idx, 'client'] = cIdx
        dfRecordAccFair.at[idx, 'numData'] = numData
        if fairness >= F_Global:
            dfRecordAccFair.at[idx, 'higher'] = "Y"
        else:
            dfRecordAccFair.at[idx, 'higher'] = "N"
        
    else: # global model
        strPrivileged = evaPrivileged
        print(f'global model accuracy on {DataType}: %.6f' % (Acc))
        print(f'global model fairness({strFairMatric}, {strPrivileged}) on {DataType}: %.6f' % (fairness))
        dfRecordAccFair.at[idx, 'dataType'] = DataType

    dfRecordAccFair.at[idx, 'Acc'] = Acc
    dfRecordAccFair.at[idx, 'fairType'] = strFairMatric
    dfRecordAccFair.at[idx, 'fairValue'] = fairness
    dfRecordAccFair.at[idx, 'Privileged'] = strPrivileged

    return dfRecordAccFair

def cal_alphaF(dfClientAccFair, client_HidCoef, client_HidBias, client_OutCoef, client_OutBias, weights_h1, bias_h1, weights_out, bias_out):

    if len(dfClientAccFair) == 0:
        alphaF_weights_h1 = weights_h1*0
        alphaF_bias_h1 = bias_h1*0
        alphaF_weights_out = weights_out*0
        alphaF_bias_out = bias_out*0
    else:
        totalF = dfClientAccFair['fairValue'].sum()
        aryFairValue = np.array(dfClientAccFair["fairValue"])
        if totalF > 0:
            ratios = np.array(aryFairValue / totalF)
        else:
            ratios = aryFairValue*0
        lsClientIdx = dfClientAccFair.index.to_list()

        weights_h1_clients = client_HidCoef[lsClientIdx]
        weights_h1_diffs = np.array([r * (w - weights_h1) for r, w in zip(ratios, weights_h1_clients)])
        alphaF_weights_h1 = weights_h1_diffs.sum(axis=0)

        bias_h1_clients = client_HidBias[lsClientIdx]
        bias_h1_diffs = np.array([r * (w - bias_h1) for r, w in zip(ratios, bias_h1_clients)])
        alphaF_bias_h1 = bias_h1_diffs.sum(axis=0)

        weights_out_clients = client_OutCoef[lsClientIdx]
        weights_out_diffs = np.array([r * (w - weights_out) for r, w in zip(ratios, weights_out_clients)])
        alphaF_weights_out = weights_out_diffs.sum(axis=0)

        bias_out_clients = client_OutBias[lsClientIdx]
        bias_out_diffs = np.array([r * (w - bias_out) for r, w in zip(ratios, bias_out_clients)])
        alphaF_bias_out = bias_out_diffs.sum(axis=0)

    return alphaF_weights_h1, alphaF_bias_h1, alphaF_weights_out, alphaF_bias_out

def cal_alphaN(dfClientAccFair, client_HidCoef, client_HidBias, client_OutCoef, client_OutBias, weights_h1, bias_h1, weights_out, bias_out):

    totalNumData = dfClientAccFair['numData'].sum()
    aryNumData = np.array(dfClientAccFair["numData"])
    ratios = np.array(aryNumData / totalNumData)
    lsClientIdx = dfClientAccFair.index.to_list()

    weights_h1_clients = client_HidCoef[lsClientIdx]
    weights_h1_diffs = np.array([r * (w - weights_h1) for r, w in zip(ratios, weights_h1_clients)])
    alphaN_weights_h1 = weights_h1_diffs.sum(axis=0)

    bias_h1_clients = client_HidBias[lsClientIdx]
    bias_h1_diffs = np.array([r * (w - bias_h1) for r, w in zip(ratios, bias_h1_clients)])
    alphaN_bias_h1 = bias_h1_diffs.sum(axis=0)

    weights_out_clients = client_OutCoef[lsClientIdx]
    weights_out_diffs = np.array([r * (w - weights_out) for r, w in zip(ratios, weights_out_clients)])
    alphaN_weights_out = weights_out_diffs.sum(axis=0)

    bias_out_clients = client_OutBias[lsClientIdx]
    bias_out_diffs = np.array([r * (w - bias_out) for r, w in zip(ratios, bias_out_clients)])
    alphaN_bias_out = bias_out_diffs.sum(axis=0)

    return alphaN_weights_h1, alphaN_bias_h1, alphaN_weights_out, alphaN_bias_out