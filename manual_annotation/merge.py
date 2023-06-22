import pandas as pd

annotators = ['Daen', 'Lin', 'Ronald']
files = [n+'/elan_annotations.csv' for n in annotators]
cols = ['tier', 'beginmm', 'begin','endmm', 'end', 'diffmm', 'diff', 'class', 'file']
df = pd.DataFrame()
for f in files:
    data = pd.read_csv(f, names=cols)
    data['annotator'] = [f.split('/')[0] for _ in range(len(data))]
    df = pd.concat([df, data], axis=0)

# df.columns = cols
df.to_csv('annotations.csv', index=False)


#double
double_files = ['double/'+n+'/elan_annotations_double.csv' for n in annotators]
# add extra annotations
for n in annotators:
    double_files.append('double/'+n+'/elan_annotations_extra.csv') 

df2 = pd.DataFrame()
for f in double_files:
    data = pd.read_csv(f, names=cols)
    data['annotator'] = [f.split('/')[1] for _ in range(len(data))]
    df2 = pd.concat([df2, data], axis=0)


# df2.columns = cols
df2.to_csv('annotations_double.csv', index=False)
