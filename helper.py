import os
import time
from datetime import datetime
from datetime import timedelta

def GetToday():
    os.system('cls||clear')
    today = datetime.today().strftime("%Y%m%d")

    return today

def GetTomorrow():
    os.system('cls||clear')
    today = datetime.today()
    tomorrow = today + timedelta(days = 1)
    tomorrow = tomorrow.strftime("%Y%m%d")

    return tomorrow

def GetTodayTime():
    todaytime = datetime.now().strftime("%Y%m%d %H.%M.%S")

    return todaytime

def CreatePlotFolder(dir_path, todayTime, strFedAlg):

    # create plot folder
    if not os.path.exists(dir_path + f'\\plot_{strFedAlg}_{todayTime}'):
        os.makedirs(dir_path + f'\\plot_{strFedAlg}_{todayTime}')
    dirname = dir_path + f'\\plot_{strFedAlg}_{todayTime}'

    return dirname

def CreateResultFolder(dir_path, todayTime, strFedAlg):

    # create plot folder
    if not os.path.exists(dir_path + f'\\result_{strFedAlg}_{todayTime}'):
        os.makedirs(dir_path + f'\\result_{strFedAlg}_{todayTime}')
    dirname = dir_path + f'\\result_{strFedAlg}_{todayTime}'

    return dirname

def CreateDatasetFolder(date):

    # create dataset folder
    if not os.path.exists(f'dataset_{date}'):
        os.makedirs(f'dataset_{date}')
    dirname = f'dataset_{date}'

    return dirname

def add_text_on_bar(ax, bars, prev_heights=None):
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if prev_heights is not None:
            height_pos = height/2 + prev_heights[i]
        else:
            height_pos = height/2

        ax.annotate('{:.2f}'.format(height*100),
                    xy=(bar.get_x() + bar.get_width() / 2, height_pos),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
def thousands_formatter(x, pos):
    return f'{x:,.0f}'

