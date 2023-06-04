import model as m
import helper
#-----------------------
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
# plt.ioff() 
from matplotlib.ticker import MultipleLocator, FuncFormatter
from sklearn.preprocessing import StandardScaler

def SplitClientTrainValDataSet(dfClientData):
    X = dfClientData.iloc[:, :-1]
    y = dfClientData.iloc[:, -1 :]

    X_test = dfClientData.iloc[:, :-1]
    y_test = dfClientData.iloc[:, -1 :]

    # validation data
    # set seed for same train/val data split
    random_seed = 5
    # X_train_orig, X_val_orig, y_train, y_val = train_test_split(X, y, random_state=random_seed, test_size = 0.2)
    X_train_orig = X

    # feature rescaling
    sc = StandardScaler()
    sc.fit(X_train_orig)
    X_test_sc = sc.transform(X_test)

    return X_test_sc, y_test


def PlotClientDataDist(listClientInfo, plot_dir_path, todayTime, datatype):

    fig, ax = plt.subplots()

    cntClient = len(listClientInfo)

    y_liPrivilegedY1 = []
    y_liPrivilegedY0 = []
    y_liUnprivilegedY1 = []
    y_liUnprivilegedY0 = []
    y_liY1inPrivileged = []
    y_liY0inPrivileged = []
    y_liY1inUnprivileged = []
    y_liY0inUnprivileged = []
    y_liCntMale = []
    y_liCntFemale = []
    y_liCntY1inPrivileged = []
    y_liCntY0inPrivileged = []
    y_liCntY1inUnprivileged = []
    y_liCntY0inUnprivileged = []

    # calculate data list for plot
    for i in range(1, cntClient + 1):

        df = listClientInfo[i-1]
        totalCnt = df.shape[0]
        totalCntPrivileged = df[df['SensitiveAttr'] == 1].shape[0]
        totalCntUnprivileged = df[df['SensitiveAttr'] == 0].shape[0]
        totalPrivilegedY1 = df[(df['SensitiveAttr'] == 1) & (df['Label'] == 1)].shape[0]
        totalPrivilegedY0 = df[(df['SensitiveAttr'] == 1) & (df['Label'] == 0)].shape[0]
        totalUnprivilegedY1 = df[(df['SensitiveAttr'] == 0) & (df['Label'] == 1)].shape[0]
        totalUnprivilegedY0 = df[(df['SensitiveAttr'] == 0) & (df['Label'] == 0)].shape[0]

        # cnt
        y_liCntY1inPrivileged.append(totalPrivilegedY1)
        y_liCntY0inPrivileged.append(totalPrivilegedY0)
        y_liCntY1inUnprivileged.append(totalUnprivilegedY1)
        y_liCntY0inUnprivileged.append(totalUnprivilegedY0)

        # ratio (label/sensAttr)
        y_liY1inPrivileged.append(round(totalPrivilegedY1/totalCntPrivileged, 4))
        y_liY0inPrivileged.append(round(totalPrivilegedY0/totalCntPrivileged, 4))
        if totalCntUnprivileged == 0:
            y_liY1inUnprivileged.append(0)
            y_liY0inUnprivileged.append(0)
        else:
            y_liY1inUnprivileged.append(round(totalUnprivilegedY1/totalCntUnprivileged, 4))
            y_liY0inUnprivileged.append(round(totalUnprivilegedY0/totalCntUnprivileged, 4))

        # ratio (label/total amount)
        y_liPrivilegedY1.append(round(totalPrivilegedY1/totalCnt, 4))
        y_liPrivilegedY0.append(round(totalPrivilegedY0/totalCnt, 4))
        y_liUnprivilegedY1.append(round(totalUnprivilegedY1/totalCnt, 4))
        y_liUnprivilegedY0.append(round(totalUnprivilegedY0/totalCnt, 4))

        # amount (sensAttr)
        y_liCntMale.append(totalCntPrivileged)
        y_liCntFemale.append(totalCntUnprivileged)


    # data set
    bar_width = 0.35
    # x = ['Client_%d'%i for i in range(1, cntClient + 1)]
    x = ['%d'%i for i in range(1, cntClient + 1)]
    x_idx = np.arange(cntClient)

    #####################################################
    # plot stacked bar chart
    ########## Plot 4 types dist ##########
    # fig, ax = plt.subplots()
    plt.bar(x, y_liCntY1inPrivileged, color='royalblue',label='Privileged, Y=1',width=bar_width)
    plt.bar(x, y_liCntY0inPrivileged, color='lightskyblue', label='Privileged, Y=0',width=bar_width, bottom=np.array(y_liCntY1inPrivileged))
    plt.bar(x, y_liCntY1inUnprivileged, color='hotpink', label='Unprivileged, Y=1',width=bar_width, bottom=np.array(y_liCntY1inPrivileged)+np.array(y_liCntY0inPrivileged))
    plt.bar(x, y_liCntY0inUnprivileged, color='pink', label='Unprivileged, Y=0',width=bar_width, bottom=np.array(y_liCntY1inPrivileged)+np.array(y_liCntY0inPrivileged)+np.array(y_liCntY1inUnprivileged))
    plt.xlabel('Client')
    plt.ylabel('Count')
    plt.title(f'Data distribution in {datatype} data')
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax.yaxis.set_major_locator(MultipleLocator(250))
    # ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}'))
   
    plt.savefig(f'{plot_dir_path}\\data_dist_in_client_{datatype}_{todayTime}.png', dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')
    # plt.show()
    # plt.clf()
    plt.close()

    #####################################################
    # plot stacked bar chart
    ########## Plot 4 types dist ##########
    fig, ax = plt.subplots()
    plt.bar(x, y_liPrivilegedY1, color='royalblue',label='Privileged, Y=1',width=bar_width)
    plt.bar(x, y_liPrivilegedY0, color='lightskyblue', label='Privileged, Y=0',width=bar_width, bottom=np.array(y_liPrivilegedY1))
    plt.bar(x, y_liUnprivilegedY1, color='hotpink', label='Unprivileged, Y=1',width=bar_width, bottom=np.array(y_liPrivilegedY1)+np.array(y_liPrivilegedY0))
    plt.bar(x, y_liUnprivilegedY0, color='pink', label='Unprivileged, Y=0',width=bar_width, bottom=np.array(y_liPrivilegedY1)+np.array(y_liPrivilegedY0)+np.array(y_liUnprivilegedY1))
    plt.xlabel('Client')
    plt.ylabel('%')
    plt.title(f'Y value(%) in {datatype} data')
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.0f}'))

    # Add numbers on top of the bars
    if cntClient <= 5:
        for i in range(len(x)):
            plt.text(x[i], y_liPrivilegedY1[i] / 2, f'{y_liPrivilegedY1[i]*100:.2f}', ha='center', va='bottom', fontsize=10)
            plt.text(x[i], y_liPrivilegedY0[i] / 2 + y_liPrivilegedY1[i], f'{y_liPrivilegedY0[i]*100:.2f}', ha='center', va='bottom', fontsize=10)
            plt.text(x[i], y_liUnprivilegedY1[i] / 2 + y_liPrivilegedY1[i] + y_liPrivilegedY0[i], f'{y_liUnprivilegedY1[i]*100:.2f}', ha='center', va='bottom', fontsize=10)
            plt.text(x[i], y_liUnprivilegedY0[i] / 2 + y_liPrivilegedY1[i] + y_liPrivilegedY0[i] + y_liUnprivilegedY1[i], f'{y_liUnprivilegedY0[i]*100:.2f}', ha='center', va='bottom', fontsize=10)
        
    plt.savefig(f'{plot_dir_path}\\y_dist_in_client_{datatype}_{todayTime}.png', dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')
    # plt.show()
    plt.close()

    ########## Plot types dist in each sensitive attribute ##########   
    fig, ax = plt.subplots()

    bars1 = plt.bar(x_idx - bar_width / 2, y_liY1inPrivileged, color='royalblue', label='Privileged, Y=1', width=bar_width)
    bars2 = plt.bar(x_idx - bar_width / 2, y_liY0inPrivileged, color='lightskyblue', label='Privileged, Y=0', width=bar_width, bottom=np.array(y_liY1inPrivileged))
    bars3 = plt.bar(x_idx + bar_width / 2, y_liY1inUnprivileged, color='hotpink', label='Unprivileged, Y=1', width=bar_width)
    bars4 = plt.bar(x_idx + bar_width / 2, y_liY0inUnprivileged, color='pink', label='Unprivileged, Y=0', width=bar_width, bottom=np.array(y_liY1inUnprivileged))

    if cntClient <= 5:
        helper.add_text_on_bar(ax, bars1)
        helper.add_text_on_bar(ax, bars2, prev_heights=y_liY1inPrivileged)
        helper.add_text_on_bar(ax, bars3)
        helper.add_text_on_bar(ax, bars4, prev_heights=y_liY1inUnprivileged)

    plt.xlabel('Client')
    plt.ylabel('%')
    plt.title(f'Y value(%) by sensitive attribute in {datatype} data')
    plt.xticks(x_idx, x)
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.0f}'))

    plt.savefig(f'{plot_dir_path}\\y_dist_in_SensAttr_client_{datatype}_{todayTime}.png', dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')
    # plt.show()
    plt.close()

    ########## Plot sensitive attribute dist ##########   
    fig, ax = plt.subplots()
    bar_width = 0.6
    opacity = 1
    space_between_clients = 1.5
    index = np.arange(0, cntClient * space_between_clients, space_between_clients)

    rects1 = plt.bar(index, y_liCntMale, bar_width, alpha=opacity, color='lightskyblue', label='Privileged')
    rects2 = plt.bar(index + bar_width , y_liCntFemale, bar_width, alpha=opacity, color='pink', label='Unprivileged')


    if cntClient <= 5:
        for i in range(len(index)):
            plt.text(index[i], y_liCntMale[i] / 2, f'{y_liCntMale[i]:,}', ha='center', va='center', fontsize=10)
            plt.text(index[i] + bar_width, y_liCntFemale[i] / 2, f'{y_liCntFemale[i]:,}', ha='center', va='center', fontsize=10)

    plt.xlabel('Client')
    plt.ylabel('Amount')
    plt.title(f'Numbers of Privileged/Unprivileged in {datatype} data')
    plt.xticks(index + bar_width / 2, x)
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax.yaxis.set_major_formatter(FuncFormatter(helper.thousands_formatter))

    # plt.tight_layout()
    plt.savefig(f'{plot_dir_path}\\gender_amounts_in_client_{datatype}_{todayTime}.png', dpi=300, bbox_extra_artists=(legend,), bbox_inches='tight')
    # plt.show()
    plt.close()

def PlotServerDataDist(df, plot_dir_path, todayTime, datatype):
    
    df.loc[:,'class'] = df['SensitiveAttr'].map({0:'Unprivileged', 1:'Privileged'}) + ', Y=' + df['Label'].astype(int).astype(str)

    labels_order = ['Privileged, Y=1', 'Privileged, Y=0', 'Unprivileged, Y=1', 'Unprivileged, Y=0']
    colors = ['royalblue', 'lightskyblue', 'hotpink', 'pink']
    class_counts = df['class'].value_counts().loc[labels_order]

    # Plot pie chart
    plt.figure(figsize=(8,8))
    wedges, texts, autotexts = plt.pie(class_counts,
                                        # labels=labels_order,
                                        autopct='%1.2f%%', startangle=140, colors=colors)
    plt.axis('equal')  
    # Adjust the title size here
    plt.title(f'Data Distribution ({datatype})', fontsize=20)

    for autotext in autotexts:
        autotext.set_size(16)

    # Add a legend
    plt.legend(wedges, labels_order,
            #    title="Classes",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize='medium')
    
    plt.savefig(f'{plot_dir_path}\\y_dist_in_{datatype}_{todayTime}.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()




