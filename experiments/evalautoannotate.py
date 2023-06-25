"""
evaluation of autoAnnotate using manual annotation data 

Measures to be collected are Cohen's Kappa

Also compute the above for visible and invisible case seperately.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats import inter_rater as irr

TABLE = 0
ROBOT = 1
TABLET = 2
ELSEWHERE = 3
UNKNOWN = 4


df = pd.read_csv('annotations_frame.csv')


def eval(path):
    fleiss_df = pd.DataFrame(columns=['file', 'case', 'kappa'])
    fleissw4_df = pd.DataFrame(columns=['file', 'case', 'kappa'])
    cohens_df = pd.DataFrame(columns=['file', 'case', 'kappa'])
    cohensw4_df = pd.DataFrame(columns=['file', 'case', 'kappa'])

    for f in df['file'].unique():
        case = list(df[df['file']==f]['case'])[0]
        anns = []

        ann = list(df[(df['file'] == f)]['class'])
        
        dfauto = pd.read_csv(path+f+'.csv')
        annauto = list(dfauto['class'].astype(int))

        anns = np.array([ann, annauto])
        annst = anns.T
        dat, cat = irr.aggregate_raters(annst)
        fk = irr.fleiss_kappa(dat, method='fleiss')

        dat = []
        for i in annst:
            i = list(i)
            new_subject = [i.count(j) for j in range(0, 4)] 
            dat.append(new_subject) if sum(new_subject) == len(i) else None
        fkw4 = irr.fleiss_kappa(dat, method='fleiss')

        ck = cohen_kappa_score(ann, annauto, labels=[TABLE, ROBOT, TABLET, ELSEWHERE, UNKNOWN])
        ckw4 = cohen_kappa_score(ann, annauto, labels=[TABLE, ROBOT, TABLET, ELSEWHERE])

        fleiss_df.loc[len(fleiss_df)] = [f, case, fk]
        cohens_df.loc[len(cohens_df)] = [f, case, ck]
        fleissw4_df.loc[len(fleissw4_df)] = [f, case, fkw4]
        cohensw4_df.loc[len(cohensw4_df)] = [f, case, ckw4]
            
    
    return fleiss_df, cohens_df, fleissw4_df, cohensw4_df


if __name__ == '__main__':
    print('extend 1\nincl, excl class4')
    fs, cs, fsw4, csw4 = eval('./l2cs_extendgaze1/')
    print(cs['kappa'].mean(), csw4['kappa'].mean())
    csv = cs[cs['case']=='visible']
    csi = cs[cs['case']=='invisible']
    csw4v = csw4[csw4['case']=='visible']
    csw4i = csw4[csw4['case']=='invisible']
    print(csv['kappa'].mean(), csw4v['kappa'].mean(), 'visible')
    print(csi['kappa'].mean(), csw4i['kappa'].mean(), 'invisible')

    print() 

    print('aggregated extend 1\nincl, excl class4')
    fs, cs, fsw4, csw4 = eval('../aggregation/aggr_l2csextendgaze1/')
    print(cs['kappa'].mean(), csw4['kappa'].mean())
    csv = cs[cs['case']=='visible']
    csi = cs[cs['case']=='invisible']
    csw4v = csw4[csw4['case']=='visible']
    csw4i = csw4[csw4['case']=='invisible']
    print(csv['kappa'].mean(), csw4v['kappa'].mean(), 'visible')
    print(csi['kappa'].mean(), csw4i['kappa'].mean(), 'invisible')

    print()

    print('extend 0\nincl, excl class4')

    fs, cs, fsw4, csw4 = eval('./l2cs_extendgaze0/')
    print(cs['kappa'].mean(), csw4['kappa'].mean())
    csv = cs[cs['case']=='visible']
    csi = cs[cs['case']=='invisible']
    csw4v = csw4[csw4['case']=='visible']
    csw4i = csw4[csw4['case']=='invisible']
    print(csv['kappa'].mean(), csw4v['kappa'].mean(), 'visible')
    print(csi['kappa'].mean(), csw4i['kappa'].mean(), 'invisible')
    print('aggregated extend 0\nincl, excl class4')

    fs, cs, fsw4, csw4 = eval('../aggregation/aggr_l2csextendgaze0/')
    print(cs['kappa'].mean(), csw4['kappa'].mean())
    csv = cs[cs['case']=='visible']
    csi = cs[cs['case']=='invisible']
    csw4v = csw4[csw4['case']=='visible']
    csw4i = csw4[csw4['case']=='invisible']
    print(csv['kappa'].mean(), csw4v['kappa'].mean(), 'visible')
    print(csi['kappa'].mean(), csw4i['kappa'].mean(), 'invisible')
    