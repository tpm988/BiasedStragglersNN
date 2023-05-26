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


def ModelEval(t, serverDataType, aryCoef, fInterception, dfServerData):

    X_train, y_train = c.SplitClientTrainValDataSet(dfServerData)
    model = m.CreateModel(aryCoef, fInterception)
    # model.predict(X_train)
    Acc = model.score(X_train, y_train)
    print(f"global model accuracy on {serverDataType} data: %.6f " % (Acc))

    # return sensitive attribute and prediction result
    df = dfServerData[['SensitiveAttr', 'Label']]
    df.insert(0, "iter", t, True)
    cntdfCol = len(df.columns)
    df.insert(cntdfCol, "Prediction", model.predict(X_train), True) # add Prediction to last column

    return Acc, df

def calFairness(t, bIsClient, DataType, strFairMatric, aryCoef, fInterception, dfServerData, dfRecordAccFair, F_Global = 0):

    dfRecordAccFair.loc[len(dfRecordAccFair.index), 'iter'] = t
    idx = dfRecordAccFair[dfRecordAccFair['iter'] == t].index[-1]

    # create model and predict
    Acc, df = ModelEval(t, DataType, aryCoef, fInterception, dfServerData)

    # evaluate fairness
    if strFairMatric == "SP":
        fairness = m.calSP(df)
    elif strFairMatric == "EO":
        fairness = m.calEO(df)
    elif strFairMatric == "EQO":
        fairness = m.calEQO(df)

    if bIsClient:
        dfRecordAccFair.at[idx, 'client'] = DataType
        if fairness >= F_Global:
            dfRecordAccFair.at[idx, 'higher'] = "Y"
        else:
            dfRecordAccFair.at[idx, 'higher'] = "N"
    else:
        dfRecordAccFair.at[idx, 'dataType'] = DataType
    dfRecordAccFair.at[idx, 'Acc'] = Acc
    dfRecordAccFair.at[idx, 'fairType'] = strFairMatric
    dfRecordAccFair.at[idx, 'fairValue'] = fairness

    return dfRecordAccFair

def cal_alphaF(cntFeature, dfClientAccFair, dfClientCoef, dfGlobalCoef):

    alphaF = []
    totalF = dfClientAccFair['fairValue'].sum()
    arycol = ['intercept'] + ['coef_x_%d' %i for i in range(1,cntFeature+1)]

    for i in range(len(arycol)):
        col = arycol[i]
        alphaF.append(np.dot((dfClientCoef[col] - dfGlobalCoef[col].values[0]), (dfClientAccFair['fairValue'] / totalF)))

    return np.array(alphaF)

def cal_alphaN(cntFeature, dfClientCoef, dfGlobalCoef):

    alphaN = []
    totalNumData = dfClientCoef['numData'].sum()
    arycol = ['intercept'] + ['coef_x_%d' %i for i in range(1,cntFeature+1)]

    for i in range(len(arycol)):
        col = arycol[i]
        alphaN.append(np.dot((dfClientCoef[col] - dfGlobalCoef[col].values[0]),(dfClientCoef['numData'] / totalNumData)))

    return np.array(alphaN)