import helper
#-----------------------
import os
import warnings
import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from sklearn.utils import shuffle
from sklearn.exceptions import ConvergenceWarning
from pandas.core.common import SettingWithCopyWarning

# Ignore ConvergenceWarning and FutureWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# pd.set_option('mode.chained_assignment', None)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
os.system('cls||clear')

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


################################################
# create synthetic dataset

def generate_synthetic_hetero_dataset(alpha, num_clients, num_classes, mean_samples_per_client, stddev_samples_per_client):

    print("------------Execute (generate_heterogeneous_data)------------")

    data_dist_per_client = get_hetero_data_dist_per_client(alpha, num_clients, num_classes, mean_samples_per_client, stddev_samples_per_client)
    
    dfTrain, dfVal, dfTest = create_synthetic_data(num_clients, num_classes, data_dist_per_client)

    print("------------Done (generate_heterogeneous_data)------------")

    return dfTrain, dfVal, dfTest

def create_synthetic_data(num_clients, num_classes, data_dist):

    print("------------Execute (create_synthetic_data)------------")

    rTrain, rTest, rVal = 0.6, 0.2, 0.2 # need sum to 1.0

    numKeyAttribute = 8
    numNormalAttribute = 6
    colSensitiveAttr = "SensitiveAttr"
    colLabel = "Label"

    all_client_data_train = []
    all_data_test = []
    all_data_val = []

    for i in range(1, num_clients + 1):
        client_data_dist = data_dist[i-1]

        client_data_train = []

        for j in range(num_classes):

            if j == 0: # Unprivileged; Y = 1
                datarows_train = generateDatarow(numKeyAttribute, numNormalAttribute, group = 0, label = 1, amount = client_data_dist[j])
                datarows_test = generateDatarow(numKeyAttribute, numNormalAttribute, group = 0, label = 1, amount = int(client_data_dist[j]*(rTest/rTrain)))
                datarows_val = generateDatarow(numKeyAttribute, numNormalAttribute, group = 0, label = 1, amount = int(client_data_dist[j]*(rVal/rTrain)))
                # first time: append
                # client_data_train.append(datarows_train)
                # all_data_test.append(datarows_test)
                # all_data_val.append(datarows_val)
                client_data_train = datarows_train
                if i == 1:
                    all_data_test = datarows_test
                    all_data_val = datarows_val
                else:
                    all_data_test = np.vstack((all_data_test, datarows_test))
                    all_data_val = np.vstack((all_data_val, datarows_val))

            elif j == 1: # Privileged; Y = 1
                datarows_train = generateDatarow(numKeyAttribute, numNormalAttribute, group = 1, label = 1, amount = client_data_dist[j])
                datarows_test = generateDatarow(numKeyAttribute, numNormalAttribute, group = 1, label = 1, amount = int(client_data_dist[j]*(rTest/rTrain)))
                datarows_val = generateDatarow(numKeyAttribute, numNormalAttribute, group = 1, label = 1, amount = int(client_data_dist[j]*(rVal/rTrain)))
            elif j == 2: # Privileged; Y = 0
                datarows_train = generateDatarow(numKeyAttribute, numNormalAttribute, group = 1, label = 0, amount = client_data_dist[j])
                datarows_test = generateDatarow(numKeyAttribute, numNormalAttribute, group = 1, label = 0, amount = int(client_data_dist[j]*(rTest/rTrain)))
                datarows_val = generateDatarow(numKeyAttribute, numNormalAttribute, group = 1, label = 0, amount = int(client_data_dist[j]*(rVal/rTrain)))
            else: # j == 3 # Unprivileged; Y = 0
                datarows_train = generateDatarow(numKeyAttribute, numNormalAttribute, group = 0, label = 0, amount = client_data_dist[j])
                datarows_test = generateDatarow(numKeyAttribute, numNormalAttribute, group = 0, label = 0, amount = int(client_data_dist[j]*(rTest/rTrain)))
                datarows_val = generateDatarow(numKeyAttribute, numNormalAttribute, group = 0, label = 0, amount = int(client_data_dist[j]*(rVal/rTrain)))
            
            # after first time: hstack
            if j > 0:
                client_data_train = np.vstack((client_data_train, datarows_train))
                all_data_test = np.vstack((all_data_test, datarows_test))
                all_data_val = np.vstack((all_data_val, datarows_val))

        # collect all client training datasets
        all_client_data_train.append(client_data_train)

    # Create the DataFrame
    column_names = [f'col_{i}' for i in range(1, numKeyAttribute + numNormalAttribute + 1)]
    column_names.extend([colSensitiveAttr, colLabel])  # Names for the additional columns

    # write testing data
    dfVal = pd.DataFrame(all_data_val, columns=column_names)
    dfVal = shuffle(dfVal)
    dfVal.to_csv(f'{dirname}\\validation_data.csv', index=False)
    print(f'Total number of Validation datasets: {dfVal.shape[0]}')

    dfTest = pd.DataFrame(all_data_test, columns=column_names)
    dfTest = shuffle(dfTest)
    dfTest.to_csv(f'{dirname}\\testing_data.csv', index=False)
    print(f'Total number of Testing datasets: {dfTest.shape[0]}')

    dfTrain = []
    for i in range(1, num_clients + 1):
        df = pd.DataFrame(all_client_data_train[i-1], columns=column_names)
        df = shuffle(df)
        df.to_csv(f'{dirname}\\training_data_client_{i}.csv', index=False)
        dfTrain.append(df)
        print(f'client {i} training data {df.shape}')

    print("------------Done (create_synthetic_data)------------")

    return dfTrain, dfVal, dfTest

def generateDatarow(numKeyAttribute, numNormalAttribute, group, label, amount,):

    ############## for Key Attribute ##############
    if label == 1:
        probabilities = [0.7, 0.3]
    elif (label == 0) and (group == 0):
        probabilities = [0.5, 0.5]
    else:
        probabilities = [0.3, 0.7]

    for i in range(numKeyAttribute):
        # Create two different distributions
        distribution1 = np.random.uniform(9, 10, amount)
        distribution2 = np.random.uniform(0, 1, amount)

        # Choose between the two distributions according to the specified probabilities
        choices = np.random.choice([0, 1], size=amount, p=probabilities)

        # Use the choices to select from the two distributions
        column_data = np.where(choices, distribution2, distribution1)

        if i == 0:
            datas = column_data
        else:
            datas = np.vstack((datas, column_data))

    ############## for Normal Attribute ##############
    for i in range(numNormalAttribute):
        column_data = np.clip(np.random.normal(5, 1.66, amount), 0, 10)
        datas = np.vstack((datas, column_data))

    row_values = datas.T

    # Create arrays for the new columns
    if group == 0:
        column1 = np.zeros((row_values.shape[0], 1))
    else:
        column1 = np.ones((row_values.shape[0], 1))

    if label == 0:
        column2 = np.zeros((row_values.shape[0], 1))
    else:
        column2 = np.ones((row_values.shape[0], 1))

    # Add new columns to the array
    row_values = np.c_[row_values, column1, column2]

    return row_values

def get_hetero_data_dist_per_client(alpha, num_clients, num_classes, mean_samples_per_client, stddev_samples_per_client):
    
    print("------------Execute (get_hetero_data_dist_per_client)------------")

    # Generate the total number of samples for each client from a Gaussian distribution
    total_samples_per_client = np.random.normal(loc=mean_samples_per_client, scale=stddev_samples_per_client, size=num_clients).astype(int)
    print("total_samples_per_client: ")
    print(total_samples_per_client)

    # Ensure positive number of samples for each client
    total_samples_per_client = np.maximum(total_samples_per_client, 0)
    if (total_samples_per_client.min() <= 0):
        generate_synthetic_hetero_dataset(alpha, num_clients, num_classes, mean_samples_per_client, stddev_samples_per_client)
    
    data_per_client = []
    for client_samples in total_samples_per_client:
        # Generate class proportions for this client from a Dirichlet distribution
        class_proportions = np.random.dirichlet([alpha] * num_classes)
        class_proportions = np.sort(class_proportions)

        # Multiply the proportions by the total number of samples to get the number of samples per class
        samples_per_class = (class_proportions * client_samples).astype(int)

        # Correct any rounding errors to make sure the total number of samples is correct
        if np.sum(samples_per_class) < client_samples:
            samples_per_class[np.argmax(class_proportions)] += client_samples - np.sum(samples_per_class)
        
        data_per_client.append(samples_per_class)

    # sort the data_per_client
    data_per_client = np.array(data_per_client)
    column_sums = data_per_client.sum(axis=0)
    sort_indices = column_sums.argsort() # find column index by ascending column sum
    data_dist_per_client = data_per_client[:, sort_indices]

    print("data_dist_per_client: ")
    print(data_dist_per_client)
    
    print("------------Done (get_hetero_data_dist_per_client)------------")

    return data_dist_per_client

# create synthetic dataset
################################################

#################################################################
# For existed dataset used

def split_existed_to_train_val_test(today_path, strFairMatric, run, rTrain, rVal, rTest, alpha, num_clients):

    todayTime = helper.GetTodayTime()
    # create dataset folder
    dirname = helper.CreateDatasetFolder(today_path, strFairMatric, run)

    f = open(dirname + f'\\output_dataset_{todayTime}.out', 'w')
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f)

    start_time = time.time()
    os.system('cls||clear')
    # dataset = 'adult_used.csv'
    dataset = 'law.csv'

    print("------------Execute (split_train_val_test)------------")
    print(f"rTrain: {rTrain}, rVal: {rVal}, rTest: {rTest}, alpha: {alpha}, num_clients: {num_clients}")

    # Load dataset

    df = pd.read_csv(dataset)
    df = shuffle(df)
    # df.rename({'sex_Male': 'SensitiveAttr', 'income_>50K': 'Label'}, axis=1, inplace=True)

    df['race'] = df['race'].map({'White': 1, 'Non-White': 0})
    df['pass_bar'] = df['pass_bar'].astype(int)
    df.rename({'race': 'SensitiveAttr', 'pass_bar': 'Label'}, axis=1, inplace=True)

    numDataRow = df.shape[0]
    numDataRowOfVal = int(numDataRow*rVal)
    numDataRowOfTest = int(numDataRow*rTest)
    print(f'Total number of data row in {dataset}: {numDataRow}')

    dfVal = df[0:numDataRowOfVal]
    dfTest = df[numDataRowOfVal:(numDataRowOfVal+numDataRowOfTest)]
    dfTrain = df[(numDataRowOfVal+numDataRowOfTest):]
    print(f'Total number of Training datasets: {dfTrain.shape[0]}')
    print(f'Total number of Testing datasets: {dfTest.shape[0]}')
    print(f'Total number of Validation datasets: {dfVal.shape[0]}')

    # Save Validation and Testing dataset to local
    dfTrain.to_csv(f'{dirname}\\training_data.csv', index=False)
    dfVal.to_csv(f'{dirname}\\validation_data.csv', index=False)
    dfTest.to_csv(f'{dirname}\\testing_data.csv', index=False)

    # Split Training data into private client dataset
    # client_datasets, num_clients = split_existed_dataset_in_heterogeneity(dfTrain, alpha, num_clients, dirname)
    splitCustomizedDataSet(dirname, dfTrain, alpha, num_clients)

    print("------------Done (split_train_val_test)------------")

    # At the end of the script, restore stdout and close the file
    sys.stdout = original
    f.close()

    return dirname

def splitCustomizedDataSet(dirname, df, alpha, num_clients):

    start_time = time.time()
    os.system('cls||clear')
    today = helper.GetToday()
    random_seed = 5

    # df = pd.read_csv(f'{dirname}\\training_data.csv', index_col=False) # total: 45201
    # df = df.sample(frac=1, random_state = 46).reset_index(drop=True)
    df = shuffle(df) 
    numDataRow = df.shape[0]

    dfMY1 = df[(df['SensitiveAttr'] == 1) & (df['Label'] == 1)]
    dfMY0 = df[(df['SensitiveAttr'] == 1) & (df['Label'] == 0)]
    dfFY1 = df[(df['SensitiveAttr'] == 0) & (df['Label'] == 1)]
    dfFY0 = df[(df['SensitiveAttr'] == 0) & (df['Label'] == 0)]

    cntdfMY1 = dfMY1.shape[0]
    cntdfMY0 = dfMY0.shape[0]
    cntdfFY1 = dfFY1.shape[0]
    cntdfFY0 = dfFY0.shape[0]

    idxMY1Start = 0
    idxMY0Start = 0
    idxFY1Start = 0
    idxFY0Start = 0

    # number of element in array = number of client model
    ratio = np.random.dirichlet([alpha] * np.ones(num_clients), 4)
    print('ratio :')
    print(ratio)

    # ratioMY1 = np.random.dirichlet([alpha] * num_clients)
    # ratioMY0 = np.random.dirichlet([alpha] * num_clients)
    # ratioFY1 = np.random.dirichlet([alpha] * num_clients)
    # ratioFY0 = np.random.dirichlet([alpha] * num_clients)

    ratioMY1 = ratio[0]
    ratioMY0 = ratio[1]
    ratioFY1 = ratio[2]
    ratioFY0 = ratio[3]

    lsRatio = list(zip(ratioMY1, ratioMY0, ratioFY1, ratioFY0))

    print(f'Total number of data row in dataset: {numDataRow}')
    print(f'Total number of clients datasets: {len(lsRatio)}')
    print ('----------------(start splitting client datasets from adult_used.csv)----------------')

    for i in range(1, len(lsRatio)+1):
        idxMY1End = int(cntdfMY1*lsRatio[i-1][0])
        idxMY0End = int(cntdfMY0*lsRatio[i-1][1])
        idxFY1End = int(cntdfFY1*lsRatio[i-1][2])
        idxFY0End = int(cntdfFY0*lsRatio[i-1][3])

        if i == len(lsRatio):
            dfTrain = pd.concat([dfMY1[idxMY1Start :],
                                dfMY0[idxMY0Start :],
                                dfFY1[idxFY1Start :],
                                dfFY0[idxFY0Start :],
                                ], axis=0)
        else:
            dfTrain = pd.concat([dfMY1[idxMY1Start : idxMY1Start + idxMY1End],
                                dfMY0[idxMY0Start : idxMY0Start + idxMY0End],
                                dfFY1[idxFY1Start : idxFY1Start + idxFY1End],
                                dfFY0[idxFY0Start : idxFY0Start + idxFY0End],
                                ], axis=0)

        idxMY1Start = idxMY1Start + idxMY1End
        idxMY0Start = idxMY0Start + idxMY0End
        idxFY1Start = idxFY1Start + idxFY1End
        idxFY0Start = idxFY0Start + idxFY0End

        dfTrain = shuffle(dfTrain)
        dfTrain.to_csv(f'{dirname}\\training_data_client_{i}.csv', index=False)
        print(f'client {i} training data {dfTrain.shape}')
    
    end_time = time.time()
    print (f'----------------(finish splitting client datasets took {end_time - start_time})----------------')

def split_existed_dataset_in_heterogeneity(df=pd.DataFrame(), alpha=0.5, num_clients=4, dirname=""):

    print("------------Execute (split_existed_dataset_in_heterogeneity)------------")

    if(df.shape[0] == 0):
        today = "20230523"
        dirname = f'dataset_{today}'
        df = pd.read_csv(dirname + '\\training_data.csv') 

    # Define four classes.
    df.loc[:, 'class'] = df['SensitiveAttr'].astype(str) + df['Label'].astype(str)
    # Map 'class' values to indices
    class_to_index = {'00': 0, '01': 1, '10': 2, '11': 3}
    df['class_index'] = df['class'].map(class_to_index)

    # Shuffle the DataFrame. This ensures that samples drawn from the DataFrame are random.
    df = shuffle(df)

    # Use a Dirichlet distribution to determine the proportion of each class for each client.
    class_proportions = np.random.dirichlet([alpha]*4, size=num_clients) # 4 refers to 4 classes: S0L0, S0L1, S1L0, S1L1

    # Save initial class sizes
    initial_class_sizes = df['class_index'].value_counts().to_dict()

    # Determine the number of samples each client should have based on a Gaussian distribution
    total_samples = df.shape[0]
    mean_samples_per_client = total_samples / num_clients
    stddev_samples_per_client = mean_samples_per_client * 0.001  # arbitrary, can be adjusted
    client_samples = np.random.normal(loc=mean_samples_per_client, 
                                    scale=stddev_samples_per_client, 
                                    size=num_clients)
    client_samples = np.round(client_samples).astype(int)

    # Ensure the total number of samples equals the total number of rows in df
    client_samples[-1] = client_samples[-1] + (total_samples - client_samples.sum())
    print(client_samples)

    # Check no client has negative sample amount
    if (client_samples.min() <= 0):
        print("Negative sample amount is detected:")
        print(client_samples)
        print("Re-split client dataset!")
        split_existed_dataset_in_heterogeneity(df, alpha, num_clients, dirname)

    client_datasets = []
    for i in range(num_clients):
        client_df = pd.DataFrame(columns=df.columns)

        for class_index in range(4):
            class_df = df[df['class_index'] == class_index]
            if i < num_clients - 1:
                n_samples = min(int(client_samples[i] * class_proportions[i, class_index]), len(class_df))
            else:
                # For the last client, assign all remaining data
                n_samples = len(class_df)
            samples = class_df.sample(n_samples, replace=False)
            client_df = pd.concat([client_df, samples])
            df = df.drop(samples.index)

        if client_df.shape[0] == 0:
            print("0 client data amount is detected:")
            print("Re-split client dataset!")
            split_existed_dataset_in_heterogeneity(df, alpha, num_clients, dirname)

        client_df = shuffle(client_df)
        client_datasets.append(client_df)

    # Remove redundant columns
    [df.drop(['class','class_index'], axis = 1, inplace = True) for df in client_datasets]

    # Save client dataset to local
    for i in range(1, num_clients + 1):
        client_datasets[i-1].to_csv(f'{dirname}\\training_data_client_{i}.csv', index=False)
        print(f'client {i} training data {client_datasets[i-1].shape}')
    
    print("------------Done (split_existed_dataset_in_heterogeneity)------------")

    return client_datasets, num_clients

# For existed dataset used
#################################################################

#################################################################
# Plot distribution

def plot_dirichlet_histogram(alpha, num_clients, n_samples=5000):
    samples = np.random.dirichlet([alpha]*num_clients, size=n_samples)

    fig, axs = plt.subplots(num_clients, figsize=(8, 6))

    for i in range(num_clients):
        axs[i].hist(samples[:, i], bins=50, alpha=0.5, color='blue')
        axs[i].set_title(f'Histogram of probabilities for Class {i+1}')

    fig.suptitle(f'Dirichlet({alpha}, {alpha}, {alpha})', y=1.05)
    plt.tight_layout()
    plt.show()
    plt.show()

def plot_pie_distribution():
    # Data to plot
    labels = ['validation (on server)', 'test', 'train (in clients)']
    sizes = [20, 20, 60]
    colors = ['lightgreen', 'pink', 'lightskyblue']
    explode = (0.05, 0.05, 0.05)  # explode 1st slice

    # Plot
    plt.figure(figsize=(6,6))
    patches, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
    autopct='%1.1f%%', shadow=False, startangle=140)

    # Increase font size of pie chart labels and autopct
    for text in texts:
        text.set_fontsize(14)
    for autotext in autotexts:
        autotext.set_fontsize(14)

    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Train, Validation, and Test Data Distribution')

    plt.savefig(f'Exp_setup.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot distribution
#################################################################

# example usage:
alpha = 3
num_clients = 10
num_classes = 4
mean_samples_per_client = 2000
stddev_samples_per_client = 300

# dfTrain, dfVal, dfTest = generate_synthetic_hetero_dataset(alpha, num_clients, num_classes, mean_samples_per_client, stddev_samples_per_client)


# split_existed_to_train_val_test(rTrain=0.6, rVal=0.2, rTest=0.2, alpha=1, num_clients=10)


# # example usage:
# plot_dirichlet_histogram(alpha=2000, num_clients=3)
# plot_dirichlet_histogram(alpha=0.5, num_clients=3)


# split_existed_to_train_val_test(rTrain=0.6, rVal=0.2, rTest=0.2, alpha=1, num_clients=10)
# client_datasets = split_existed_dataset_in_heterogeneity(alpha=1, num_clients=9)



################################################

