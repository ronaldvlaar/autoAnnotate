
# Inspired by https://towardsdatascience.com/inter-rater-agreement-kappas-69cd8b91ff75
from sklearn.metrics import cohen_kappa_score
import pandas as pd

rater1 = ['no', 'no', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'no']
rater2 = ['yes', 'no', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes']
print(cohen_kappa_score(rater1, rater2, labels=['yes', 'no']))


annotations_double = pd.read_csv('annotations_double_formatted_gaps_removed.csv')
annotations = pd.read_csv('annotations_formatted_gaps_removed.csv')

dfkappa = pd.DataFrame(columns=['file', 'ckappa'])
for f in annotations_double['file'].unique():
    rater1 = list(annotations[annotations['file'] == f]['class'])
    rater2 = list(annotations_double[annotations_double['file'] == f]['class'])
    cohenk= cohen_kappa_score(rater1, rater2, labels=[0,1,2,3,4])
    dfkappa.loc[len(dfkappa)] = [f, cohenk]

print(dfkappa['ckappa'].mean())
dfkappa.to_csv('ckappa_scores.csv')