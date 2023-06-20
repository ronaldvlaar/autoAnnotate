import pandas as pd

annotations = pd.read_csv('annotations_frame.csv')
annotations_double = pd.read_csv('annotations_double_frame.csv')

for f in annotations['file'].unique():
    annotations[annotations['file'] == f].to_csv('frame_files/'+f+'.csv')

for f in annotations_double['file'].unique():
    annotations_double[annotations_double['file'] == f].to_csv('frame_files/'+f+'double.csv')

