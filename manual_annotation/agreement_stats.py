import pandas as pd

# dfc = pd.read_csv('ckappa_scores.csv')

dfcw4= pd.read_csv('ckappa_scores_without_class4.csv')


# Combinations
# Lin - Ronald 

dflr = dfcw4[((dfcw4['annotator'] == 'Ronald') & (dfcw4['2ndannotator'] == 'Lin')) | ((dfcw4['annotator'] == 'Lin') & (dfcw4['2ndannotator'] == 'Ronald'))]
print('Lin and Ronald')
print('count', dflr.count()[0])
print('avg score', dflr['ckappa'].mean())
print('min: {m:.3f}, max: {n:.3f}, std: {o:.3f}'.format(m=dflr['ckappa'].min(), n=dflr['ckappa'].max(), o=dflr['ckappa'].std()))
print()

# Lin - Daen
dfld = dfcw4[((dfcw4['annotator'] == 'Lin') & (dfcw4['2ndannotator'] == 'Daen')) | ((dfcw4['annotator'] == 'Daen') & (dfcw4['2ndannotator'] == 'Lin'))]
print('Lin and Daen')
print('count', dfld.count()[0])
print('avg score', dfld['ckappa'].mean())
print('min: {m:.3f}, max: {n:.3f}, std: {o:.3f}'.format(m=dfld['ckappa'].min(), n=dfld['ckappa'].max(), o=dfld['ckappa'].std()))

print()
# Daen - Ronald

dfdr = dfcw4[((dfcw4['annotator'] == 'Daen') & (dfcw4['2ndannotator'] == 'Ronald')) | ((dfcw4['annotator'] == 'Ronald') & (dfcw4['2ndannotator'] == 'Daen'))]
print('Daen and Ronald')
print('count', dfdr.count()[0])
print('avg_score', dfdr['ckappa'].mean())
print('min: {m:.3f}, max: {n:.3f}, std: {o:.3f}'.format(m=dfdr['ckappa'].min(), n=dfdr['ckappa'].max(), o=dfdr['ckappa'].std()))

print()

nextannotations_daen = dfcw4[(dfcw4['annotator'] != 'Daen') & (dfcw4['2ndannotator'] != 'Daen')]['file'].unique()
nextannotations_ronald = dfcw4[(dfcw4['annotator'] != 'Ronald') & (dfcw4['2ndannotator'] != 'Ronald')]['file'].unique()
nextannotations_lin = dfcw4[(dfcw4['annotator'] != 'Lin') & (dfcw4['2ndannotator'] != 'Lin')]['file'].unique()

print('To make sure all 21 videos have 3 annotators')
print('Daen annotates:')
for i in nextannotations_daen:
    print (i)
print()
print('Ronald annotates:')
for i in nextannotations_ronald:
    print (i)
print()
print('Lin annotates:')
for i in nextannotations_lin:
    print (i)


print('Files Lin selected from initial annotation for triple annotation')
print('Initial annotation from')
namecountRonaldfirst = len(dfcw4[dfcw4['annotator']=='Ronald']['file'].unique())
namecountDaenfirst = len(dfcw4[dfcw4['annotator']=='Daen']['file'].unique())
namecountLinfirst = len((dfcw4[dfcw4['annotator']=='Lin']['file'].unique()))
namecountRonaldsecond = len(dfcw4[dfcw4['2ndannotator']=='Ronald']['file'].unique())
namecountDaensecond = len(dfcw4[dfcw4['2ndannotator']=='Daen']['file'].unique())
namecountLinsecond = len((dfcw4[dfcw4['2ndannotator']=='Lin']['file'].unique()))

print('Ronald:', namecountRonaldfirst)
print('Daen:',  namecountRonaldfirst)
print('Lin:', namecountLinfirst)



print(namecountRonaldfirst+namecountRonaldsecond)
print(namecountDaenfirst+namecountDaensecond)
print(namecountLinfirst+namecountLinsecond)

