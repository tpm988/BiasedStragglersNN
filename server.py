import numpy as np
import model as m
import client as c

######(Aggregate global model coefficient)######
def FedAvg(iterGlobal, cntFeature, dfClientModelCoefficient):

    totalNumData = dfClientModelCoefficient['numData'].sum()
    aryGlobalCoef = []
    fGlobalInterception = dfClientModelCoefficient['intercept'].dot(dfClientModelCoefficient['numData'] / totalNumData)

    for i in range(1,cntFeature+1):
        col = 'coef_x_%d' %i
        aryGlobalCoef.append(dfClientModelCoefficient[col].dot(dfClientModelCoefficient['numData'] / totalNumData))
    
    return aryGlobalCoef, fGlobalInterception


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

def calFairness(t, bIsClient, DataType, strFairMatric, model, dfServerData, dfRecordAccFair, F_Global = 0, cIdx=0, numData=0):

    dfRecordAccFair.loc[len(dfRecordAccFair.index), 'iter'] = t
    idx = dfRecordAccFair[dfRecordAccFair['iter'] == t].index[-1]

    # create model and predict
    Acc, df = ModelEval(t, model, dfServerData)

    # evaluate fairness
    if strFairMatric == "SP":
        fairness = m.calSP(df)
    elif strFairMatric == "EO":
        fairness = m.calEO(df)
    elif strFairMatric == "EQO":
        fairness = m.calEQO(df)

    if bIsClient:
        print(f'client {cIdx} accuracy on {DataType}: %.6f' % (Acc))
        print(f'client {cIdx} fairness({strFairMatric}) on {DataType}: %.6f' % (fairness))
        dfRecordAccFair.at[idx, 'client'] = cIdx
        dfRecordAccFair.at[idx, 'numData'] = numData
        if fairness >= F_Global:
            dfRecordAccFair.at[idx, 'higher'] = "Y"
        else:
            dfRecordAccFair.at[idx, 'higher'] = "N"
        
    else:
        print(f'global model accuracy on {DataType}: %.6f' % (Acc))
        print(f'global model fairness({strFairMatric}) on {DataType}: %.6f' % (fairness))
        dfRecordAccFair.at[idx, 'dataType'] = DataType

    dfRecordAccFair.at[idx, 'Acc'] = Acc
    dfRecordAccFair.at[idx, 'fairType'] = strFairMatric
    dfRecordAccFair.at[idx, 'fairValue'] = fairness

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