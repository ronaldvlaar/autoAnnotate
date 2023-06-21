
# Inspired by https://towardsdatascience.com/inter-rater-agreement-kappas-69cd8b91ff75
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import numpy as np

TABLE = 0
ROBOT = 1
TABLET = 2
ELSEWHERE = 3
UNKNOWN = 4

annotations_double = pd.read_csv('annotations_double_frame.csv')
annotations = pd.read_csv('annotations_frame.csv')

dfkappa = pd.DataFrame(columns=['file', 'annotator', '2ndannotator', 'case', 'ckappa'])
dfkappa_without_unknown = pd.DataFrame(columns=['file', 'annotator', '2ndannotator', 'case', 'ckappa'])
for f in annotations_double['file'].unique():
    fcase = list(annotations[annotations['file'] == f]['case'])[0]
    annotator = list(annotations[annotations['file'] == f]['annotator'])[0]
    annotator2nd = list(annotations_double[annotations_double['file'] == f]['annotator'])[0]

    rater1 = list(annotations[annotations['file'] == f]['class'])
    rater2 = list(annotations_double[annotations_double['file'] == f]['class'])

    # rater1 = np.array(rater1).astype(int)
    # rater1[rater1==4] = 3
    # rater2 = np.array(rater2).astype(int)
    # rater2[rater2==4] = 3

    cohenk= cohen_kappa_score(rater1, rater2, labels=[TABLE, ROBOT, TABLET, ELSEWHERE, UNKNOWN])
    dfkappa.loc[len(dfkappa)] = [f, annotator, annotator2nd, fcase, cohenk]
    cohenk= cohen_kappa_score(rater1, rater2, labels=[TABLE, ROBOT, TABLET, ELSEWHERE])
    dfkappa_without_unknown.loc[len(dfkappa_without_unknown)] = [f, annotator, annotator2nd, fcase, cohenk]


print('Overall')
print('avg_score:', dfkappa_without_unknown['ckappa'].mean())
print('min:', dfkappa_without_unknown['ckappa'].min())
print('max:', dfkappa_without_unknown['ckappa'].max())
print('std:', dfkappa_without_unknown['ckappa'].std())

dfkappa.to_csv('ckappa_scores.csv')
dfkappa_without_unknown.to_csv('ckappa_scores_without_class4.csv')
