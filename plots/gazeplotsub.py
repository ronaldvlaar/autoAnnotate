import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

def get_classname(id):
    if id == 0:
        return 'table'
    elif id == 1:
        return 'robot'
    elif id == 2:
        return 'tablet'
    elif id == 3:
        return 'elsewhere'
    elif id == 4:
        return 'unknown'

classes = list(range(0, 5))
filename = '33001_sessie1_taskrobotEngagement.csv'


def plot(filename, ax, ylabel='', xlabel='', title=''):
    df = pd.read_csv(filename)
    df['class'] = df['class'].astype(int)
    nr_frames = len(df.index)
    x = range(0, nr_frames)
    bars = []
    bottoms = []
    for c in classes:
        arr = 1*np.array(df['class']==c)
        # arr[arr == 0] = c
        bars.append(arr)
        bottom = np.ones(nr_frames).astype(int)*c
        bottoms.append(bottom)

    colors = ['r', 'g', 'b', 'orange', 'violet']
    for b, bot, col in zip(bars, bottoms, colors):
        ax.bar(x, height=b, width=1, bottom=bot, color=col)

    ax.hlines(classes[1:], 0, nr_frames, color='lightgrey', linestyles='solid', linewidth=0.5)
    ax.set_title(title, fontsize=8)

    ax.set_ylim((0, 5))
    ax.set_xlim((0, nr_frames))
    ax.set_yticks([i+0.5 for i in classes])
    ax.set_yticklabels([get_classname(i) for i in classes]) 
    ax.set_ylabel(ylabel, fontsize=10) 
    ax.set_xlabel(xlabel, fontsize=10) 

    return ax


if __name__ == '__main__':
    dir = '../manual_annotation/frame_files/'
    files = [f for f in listdir(dir) if isfile(join(dir, f))]

    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True, figsize=(6,3))

    files = list(filter(lambda x: '33007_sessie1' in x,files))
    print(files[0])
    ax1= plot(dir+files[0], ax1, ylabel='gaze', title='Daen')
    ax2 =plot(dir+files[2], ax2, xlabel='frame', title='Ronald')
    ax3=plot(dir+files[1], ax3, title='Lin')
    fig.tight_layout()
    plt.savefig('../manual_annotation/frame_files/'+'compgaze.png', dpi=300)

