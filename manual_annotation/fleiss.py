# https://stackoverflow.com/questions/56481245/inter-rater-reliability-calculation-for-multi-raters-data
import pandas as pd
import numpy as np
from statsmodels.stats import inter_rater as irr
import krippendorff as kd


annotations_double = pd.read_csv('annotations_double_frame.csv')
annotations = pd.read_csv('annotations_frame.csv')
df = pd.concat([annotations, annotations_double], axis=0)


fleiss = []
krippen = []
fleissw4 = []
for f in annotations_double['file'].unique():
    names = list(df[df['file'] == f]['annotator'].unique())

    anns = []
    for n in names:
        ann = list(df[(df['file'] == f)  & (df['annotator'] == n)]['class'])
        anns.append(ann)
    

    krippen.append(kd.alpha(anns, level_of_measurement='nominal'))

    annst = np.array(anns).T
    dat, cat = irr.aggregate_raters(annst)
    fleiss.append(irr.fleiss_kappa(dat, method='fleiss'))


    annst = np.array(anns).T
    dat = []

    # https://en.wikipedia.org/wiki/Fleiss'_kappa
    for i in annst:
        i = list(i)
        new_subject = [i.count(j) for j in range(0, 4)] 
        dat.append(new_subject) if sum(new_subject) == len(i) else None
    
    # print(dat)
    fleissw4.append(irr.fleiss_kappa(dat, method='fleiss'))

    
    # break

print(np.array(fleiss).mean())
print(np.array(krippen).mean())
print(np.array(fleissw4).mean())