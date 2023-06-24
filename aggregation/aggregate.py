
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8987791
# https://dl.acm.org/doi/pdf/10.1145/355017.355028
import pandas as pd
import math
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy import stats

file = '33002_sessie2_taskrobotEngagement'
dat = pd.read_csv(file+'.csv')
dat = dat.reset_index()
dat['class'] = dat['class'].astype(int)
annotations = pd.read_csv('../manual_annotation/annotations_frame.csv')
ann = annotations[annotations['file'] == file]['class']

vidstats = pd.read_csv('../manual_annotation/vidstat.csv')
print(vidstats)
fps = vidstats[vidstats['file'] == file]['rate'].iloc[0]
tot_frames = vidstats[vidstats['file'] == file]['frames'].iloc[0]
threshold = float(40)
angles = []


for idx, row in dat.iterrows():
    # print(row['hbottom'])
    tlx = row['hleft']
    tly = row['htop']
    brx = row['hright']
    bry = row['hbottom']

    eyex, eyey = (int(tlx+brx / 2.0), int(tly+bry / 2.0))

    slope1 = abs(eyex-row['gazey'])/abs(eyex-row['gazex'])

    if idx+1 <len(dat):
        nextrow = dat.loc[idx+1]

        # print(row['hbottom'])
        tlx = nextrow['hleft']
        tly = nextrow['htop']
        brx = nextrow['hright']
        bry = nextrow['hbottom']

        eyex, eyey = int((tlx+(brx-tlx))) / 2.0, int((tly+(bry-tly)) / 2.0)

        slope2 = abs(eyey-nextrow['gazey'])/abs(eyex-nextrow['gazex'])

        angle = math.degrees(math.atan(abs((slope1-slope2)/(1+slope1*slope2))))
        angles.append(angle)
        # print(angle)

        # if math.isnan(angle):
        #     print(angle, 'a')

angles = np.array(angles)
print(angles)
fixation_groups = list(np.where(angles > threshold)[0])
fixation_groups = np.append(fixation_groups, tot_frames)
print(fixation_groups)
class_aggregated = []
last = 0
for fixation in fixation_groups:
    if last == fixation:
        continue
    mode = 4
    if len(range(last,fixation)) >=fps:
        mode = stats.mode(dat['class'][last:fixation])[0][0]
        # mode = dat['class'].iloc[last+13]

    for _ in range(last, fixation):
        class_aggregated.append(mode)
    # print(stats.mode(dat['class'][last:fixation])[0][0])
    last=fixation

dat['class_aggr'] = class_aggregated
ck = cohen_kappa_score(dat['class'].astype(int), ann, labels=[0,1,2,3,4])
ckw4 = cohen_kappa_score(dat['class'].astype(int), ann, labels=[0,1,2,3])
print(ck, ckw4)

ck = cohen_kappa_score(dat['class_aggr'].astype(int), ann, labels=[0,1,2,3,4])
ckw4 = cohen_kappa_score(dat['class_aggr'].astype(int), ann, labels=[0,1,2,3])
print(ck, ckw4)
# print('a',angles > threshold)
