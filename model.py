import client
#-----------------------
import numpy as np
import tensorflow as tf
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import SGDClassifier

bFirst = True
bSecond = True
iIterGlobal = 1

######(Create model)######
def ModelCompile(model, lr):

    sgd = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])  

    return model   

def CreatModelNNInit(n_features, lr):

    model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(n_features,)),
            tf.keras.layers.Dense(10, activation='tanh', 
                                kernel_initializer='he_normal'), # glorot_normal, glorot_uniform
            tf.keras.layers.Dense(1, activation='sigmoid', 
                                kernel_initializer='he_normal')
        ])
    
    model = ModelCompile(model, lr)
    
    # Get weights and biases for layers
    weights_h1 = model.get_weights()[0]
    bias_h1 = model.get_weights()[1]
    weights_out = model.get_weights()[2]
    bias_out = model.get_weights()[3]
    
    return model, weights_h1, bias_h1, weights_out, bias_out

def CreatModelNN(n_features, lr, weights_h1, bias_h1, weights_out, bias_out):

    model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(n_features,)),
            tf.keras.layers.Dense(10, activation='tanh', 
                                kernel_initializer=tf.constant_initializer(weights_h1),
                                bias_initializer=tf.constant_initializer(bias_h1)),
            tf.keras.layers.Dense(1, activation='sigmoid', 
                                kernel_initializer=tf.constant_initializer(weights_out),
                                bias_initializer=tf.constant_initializer(bias_out))
        ])
    
    model = ModelCompile(model, lr)
    
    return model

def CloneModelNN(model, lr):

    weights = model.get_weights()
    model_copy = tf.keras.models.clone_model(model)
    model_copy.set_weights(weights)

    model_copy = ModelCompile(model_copy, lr)

    return model_copy

######(Train client model)######
def train_NN_SGD(iterGlobal, idxClient, dfClientData, model, lr, client_HidCoef, client_HidBias, client_OutCoef, client_OutBias):

    global iIterGlobal
    global bFirst
    global bSecond
    if iterGlobal > iIterGlobal: # within the same run: always iterGlobal >= iIterGlobal 
        iIterGlobal += 1
        bFirst = True
        bSecond = True

    if iterGlobal < iIterGlobal: # next run: initialize
        iIterGlobal = 1
        bFirst = True
        bSecond = True

    x_train, y_train = client.SplitClientTrainValDataSet(dfClientData)

    model = ModelCompile(model, lr)

    # Train the model
    history = model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=0) # , validation_data=(x_val, y_val)

    # Get the accuracy
    acc_train = history.history['accuracy'][-1]
    # validation_accuracy = history.history['val_accuracy']

    # print the training accuracy
    print("training accuracy (client %s) : %.6f " % (str(idxClient), acc_train))

    # Retrieve weights and biases
    client_weights_h1 = model.get_weights()[0]
    client_bias_h1 = model.get_weights()[1]
    client_weights_out = model.get_weights()[2]
    client_bias_out = model.get_weights()[3]

    # Store in your arrays
    if bFirst:
        client_HidCoef = client_weights_h1
        client_HidBias = client_bias_h1
        client_OutCoef = client_weights_out
        client_OutBias = client_bias_out
        bFirst = False
    elif bSecond:
        client_HidCoef = np.stack((client_HidCoef, client_weights_h1))
        client_HidBias = np.stack((client_HidBias, client_bias_h1))
        client_OutCoef = np.stack((client_OutCoef, client_weights_out))
        client_OutBias = np.stack((client_OutBias, client_bias_out))
        bSecond = False
    else:
        client_HidCoef = np.concatenate((client_HidCoef, client_weights_h1[np.newaxis, :]), axis=0)
        client_HidBias = np.concatenate((client_HidBias, client_bias_h1[np.newaxis, :]))
        client_OutCoef = np.concatenate((client_OutCoef, client_weights_out[np.newaxis, :]))
        client_OutBias = np.concatenate((client_OutBias, client_bias_out[np.newaxis, :]))

    return model, client_HidCoef, client_HidBias, client_OutCoef, client_OutBias

def calSP(df):

    SP = 0
    totalPrivileged = df[df['SensitiveAttr'] == 1].shape[0]
    totalUnprivileged = df[df['SensitiveAttr'] == 0].shape[0]
    totalPrivilegedPredictY1 = df[(df['SensitiveAttr'] == 1) & (df['Prediction'] == 1)].shape[0]
    totalUnprivilegedPredictY1 = df[(df['SensitiveAttr'] == 0) & (df['Prediction'] == 1)].shape[0]

    probPrivileged = totalPrivilegedPredictY1 / totalPrivileged
    probUnprivileged = totalUnprivilegedPredictY1 / totalUnprivileged

    if (probPrivileged > 0):
        SP = round(probUnprivileged / probPrivileged, 4)

    return SP

def calEO(df):

    EO = 0
    totalPrivilegedY1 = df[(df['SensitiveAttr'] == 1) & (df['Label'] == 1)].shape[0]
    totalUnprivilegedY1 = df[(df['SensitiveAttr'] == 0) & (df['Label'] == 1)].shape[0]
    totalPrivilegedY1PredictY1 = df[(df['SensitiveAttr'] == 1) & (df['Label'] == 1) & (df['Prediction'] == 1)].shape[0]
    totalUnprivilegedY1PredictY1 = df[(df['SensitiveAttr'] == 0) & (df['Label'] == 1) & (df['Prediction'] == 1)].shape[0]

    probPrivileged = totalPrivilegedY1PredictY1 / totalPrivilegedY1
    probUnprivileged = totalUnprivilegedY1PredictY1 / totalUnprivilegedY1

    if (probPrivileged > 0):
        EO = round(probUnprivileged / probPrivileged, 4)

    return EO

def calEQO(df):

    EQO = 0
    totalPrivilegedY1 = df[(df['SensitiveAttr'] == 1) & (df['Label'] == 1)].shape[0]
    totalUnprivilegedY1 = df[(df['SensitiveAttr'] == 0) & (df['Label'] == 1)].shape[0]
    totalPrivilegedY1PredictY1 = df[(df['SensitiveAttr'] == 1) & (df['Label'] == 1) & (df['Prediction'] == 1)].shape[0]
    totalUnprivilegedY1PredictY1 = df[(df['SensitiveAttr'] == 0) & (df['Label'] == 1) & (df['Prediction'] == 1)].shape[0]

    totalPrivilegedY0 = df[(df['SensitiveAttr'] == 1) & (df['Label'] == 0)].shape[0]
    totalUnprivilegedY0 = df[(df['SensitiveAttr'] == 0) & (df['Label'] == 0)].shape[0]
    totalPrivilegedY0PredictY1 = df[(df['SensitiveAttr'] == 1) & (df['Label'] == 0) & (df['Prediction'] == 1)].shape[0]
    totalUnprivilegedY0PredictY1 = df[(df['SensitiveAttr'] == 0) & (df['Label'] == 0) & (df['Prediction'] == 1)].shape[0]

    # TP: true positive; FP: false positive
    TPPrivileged = totalPrivilegedY1PredictY1 / totalPrivilegedY1
    FPPrivileged = totalPrivilegedY0PredictY1 / totalPrivilegedY0
    TPUnprivileged = totalUnprivilegedY1PredictY1 / totalUnprivilegedY1
    FPUnprivileged = totalUnprivilegedY0PredictY1 / totalUnprivilegedY0

    # TP + FP mean
    probPrivileged = (TPPrivileged + FPPrivileged) / 2
    probUnprivileged = (TPUnprivileged + FPUnprivileged) / 2

    if (probPrivileged > 0):
        EQO = round(probUnprivileged / probPrivileged, 4)

    return EQO

def calAcc(df):
    
    totalCnt = df.shape[0]
    totalAccurateCnt = df[df['Label'] == df['Prediction']].shape[0]

    return round(totalAccurateCnt / totalCnt, 4)

