import pandas as pd
from sklearn.metrics import cohen_kappa_score

TABLE, ROBOT, TABLET, ELSEWHERE, UNKNOWN = 0, 1, 2, 3, 4
f = '33001_sessie1_taskrobotEngagement'
annl2cs = list(pd.read_csv('./l2cs_extendgaze0/'+f+'.csv')
               ['class'].astype(int))
annm4 = list(pd.read_csv('./model4_extendgaze0/'+f+'.csv')
             ['class'].astype(int))
annman = list(pd.read_csv('../manual_annotation/frame_files/' +
              f+'Ronald.csv')['class'].astype(int))

ck = cohen_kappa_score(annl2cs, annman, labels=[
                       TABLE, ROBOT, TABLET, ELSEWHERE, UNKNOWN])
ck4 = cohen_kappa_score(annl2cs, annman, labels=[
                        TABLE, ROBOT, TABLET, ELSEWHERE])


ck2 = cohen_kappa_score(annm4, annman, labels=[
                        TABLE, ROBOT, TABLET, ELSEWHERE, UNKNOWN])
ck42 = cohen_kappa_score(annm4, annman, labels=[
                         TABLE, ROBOT, TABLET, ELSEWHERE])

ck3 = cohen_kappa_score(annm4, annl2cs, labels=[
                        TABLE, ROBOT, TABLET, ELSEWHERE, UNKNOWN])
ck43 = cohen_kappa_score(annm4, annl2cs, labels=[
                         TABLE, ROBOT, TABLET, ELSEWHERE])
ck3 = cohen_kappa_score(annl2cs, annm4, labels=[
                        TABLE, ROBOT, TABLET, ELSEWHERE, UNKNOWN])
ck43 = cohen_kappa_score(annl2cs, annm4, labels=[
                         TABLE, ROBOT, TABLET, ELSEWHERE])

print(ck)
print(ck4)
print()
print(ck2)
print(ck42)
print()
print(ck3)
print(ck43)
print()
# print(annm4, annman)
