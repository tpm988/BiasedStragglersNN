import client
#-----------------------
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

######(Create model)######
def CreatModelNNInit(n_features, weights_h1, bias_h1, weights_out, bias_out):
    model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(n_features,)),
            tf.keras.layers.Dense(10, activation='tanh', 
                                kernel_initializer=tf.constant_initializer(weights_h1),
                                bias_initializer=tf.constant_initializer(bias_h1)),
            tf.keras.layers.Dense(1, activation='sigmoid', 
                                kernel_initializer=tf.constant_initializer(weights_out),
                                bias_initializer=tf.constant_initializer(bias_out))
        ])
    
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])  
    
    return model

def CloneModelNN(model):
    weights = model.get_weights()
    model_copy = tf.keras.models.clone_model(model)
    model_copy.set_weights(weights)

    return model_copy

# not used
def CreateModelNN(n_features, aryHidCoef, fHidInterception, aryOutCoef, fOutInterception):
    # create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(n_features,)),
        tf.keras.layers.Dense(10, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    if len(aryHidCoef) > 0:
        # Weights and biases for the hidden layer
        hidden_weights = np.array(aryHidCoef).reshape(n_features, -1)
        hidden_biases = np.array(fHidInterception).reshape(-1)
        model.layers[1].set_weights([hidden_weights, hidden_biases])

        # Weights and biases for the output layer
        # (you'll need to supply appropriate values for these)
        output_weights = np.array(aryOutCoef).reshape(10, 1)
        output_biases = np.array(fOutInterception).reshape(-1)
        model.layers[2].set_weights([output_weights, output_biases])

    return model

######(Train client model)######
def train_NN_SGD(iterGlobal, idxClient, n_features, dfClientData, dfClientAccFair, model, client_HidCoef, client_HidBias, client_OutCoef, client_OutBias):

    x_train, y_train = client.SplitClientTrainValDataSet(dfClientData)

    # aryHidCoef = global_HidCoef[idxClient-1][1:]
    # fHidInterception = global_HidCoef[idxClient-1][0]
    # aryOutCoef = global_OutCoef[idxClient-1][1:]
    # fOutInterception = global_OutCoef[idxClient-1][0]

    # create model
    # model = CreateModelNN(n_features, aryHidCoef, fHidInterception, aryOutCoef, fOutInterception) 
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, batch_size=10, epochs=10) # , validation_data=(x_val, y_val)

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
    if iterGlobal == 1:
        client_HidCoef = client_weights_h1
        client_HidBias = client_bias_h1
        client_OutCoef = client_weights_out
        client_OutBias = client_bias_out
    else:
        client_HidCoef = np.vstack((client_HidCoef, client_weights_h1))
        client_HidBias = np.vstack((client_HidBias, client_bias_h1))
        client_OutCoef = np.vstack((client_OutCoef, client_weights_out))
        client_OutBias = np.vstack((client_OutBias, client_bias_out))

    # copy model for evaluation
    model_client = CloneModelNN(model)

    return model_client, client_HidCoef, client_HidBias, client_OutCoef, client_OutBias

def calSP(df):

    totalPrivileged = df[df['SensitiveAttr'] == 1].shape[0]
    totalUnprivileged = df[df['SensitiveAttr'] == 0].shape[0]
    totalPrivilegedPredictY1 = df[(df['SensitiveAttr'] == 1) & (df['Prediction'] == 1)].shape[0]
    totalUnprivilegedPredictY1 = df[(df['SensitiveAttr'] == 0) & (df['Prediction'] == 1)].shape[0]

    probPrivileged = totalPrivilegedPredictY1 / totalPrivileged
    probUnprivileged = totalUnprivilegedPredictY1 / totalUnprivileged

    SP = round(probUnprivileged / probPrivileged, 4)

    return SP

def calEO(df):

    totalPrivilegedY1 = df[(df['SensitiveAttr'] == 1) & (df['Label'] == 1)].shape[0]
    totalUnprivilegedY1 = df[(df['SensitiveAttr'] == 0) & (df['Label'] == 1)].shape[0]
    totalPrivilegedY1PredictY1 = df[(df['SensitiveAttr'] == 1) & (df['Label'] == 1) & (df['Prediction'] == 1)].shape[0]
    totalUnprivilegedY1PredictY1 = df[(df['SensitiveAttr'] == 0) & (df['Label'] == 1) & (df['Prediction'] == 1)].shape[0]

    probPrivileged = totalPrivilegedY1PredictY1 / totalPrivilegedY1
    probUnprivileged = totalUnprivilegedY1PredictY1 / totalUnprivilegedY1

    EO = round(probUnprivileged / probPrivileged, 4)

    return EO

def calEQO(df):

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

    EQO = round(probUnprivileged / probPrivileged, 4)

    return EQO

def calAcc(df):
    
    totalCnt = df.shape[0]
    totalAccurateCnt = df[df['Label'] == df['Prediction']].shape[0]

    return round(totalAccurateCnt / totalCnt, 4)

