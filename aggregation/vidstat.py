import pandas as pd
from readbbtxt import readbbtxt
import cv2

datafolder = '../data_fin/'
datafilevis = 'pixel_position_vis.txt'
datafileinvis = 'pixel_position_invis_new.txt'

datavis = readbbtxt(datafolder+datafilevis)
# remove .png extension from filenames
datavis['file'] = datavis['file'].apply(lambda x: x[:-4])

datainvis = readbbtxt(datafolder+datafileinvis)
# remove .png extension from filenames
datainvis['file'] = datainvis['file'].apply(lambda x: x[:-4])

ff = '{:.3f}'.format

annotations = pd.read_csv(
    '../manual_annotation/annotations_formmatted_with_gaps.csv')


def get_stats(root, data, cols=['file', 'frames', 'rate', 'width', 'height', 'case'], case='visible'):
    vidstats, annotationstats = pd.DataFrame(columns=cols), pd.DataFrame(columns=['file', 'count', 'avg duration', 'avg duration c0', 'avg duration c1', 
                                                                                  'avg duration c2', 'avg duration c3', 'avg duration c4', 'count c0', 'count c1','count c2','count c3','count c4'])
    for f in data['file']:
        video = f+'.MP4'
        # print(root+video)
        cap = cv2.VideoCapture(root+video)
        if not cap.isOpened():
            video = f+'.mp4'
            cap = cv2.VideoCapture(root+video)
            if not cap.isOpened():
                raise IOError("Could not read the video file")

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        rec = annotations[annotations['file'] == f]
        anncount = len(rec)
        meanannd = ff(float(rec['diff'].mean()))


        rec0 = rec[rec['class'].astype(int) == 0]
        rec1 = rec[rec['class'].astype(int) == 1]
        rec2 = rec[rec['class'].astype(int) == 2]
        rec3 = rec[rec['class'].astype(int) == 3]
        rec4 = rec[rec['class'].astype(int) == 4]
        
        vidstats.loc[len(vidstats)] = [f, length, fps, width, height, case]
        annotationstats.loc[len(annotationstats)] = [f, anncount, meanannd, ff(float(rec0['diff'].mean())),ff(float(rec1['diff'].mean())),ff(float(rec2['diff'].mean())),ff(float(rec3['diff'].mean())),ff(float(rec4['diff'].mean())),
                                                     len(rec0),len(rec1),len(rec2),len(rec3),len(rec4)]

    return vidstats, annotationstats


vidstatsvis, annotationsstat1 = get_stats(
    '../all_vids/visible_with_bounding_boxes/', datavis, case='visible')
vidstatsinv, annotationsstat2 = get_stats(
    '../all_vids/invisible_with_bounding_boxes/', datainvis, case='invisible')
vidstat = pd.concat([vidstatsvis, vidstatsinv], axis=0)
annostat = pd.concat([annotationsstat1, annotationsstat2], axis=0)
vidstat.to_csv('vidstat.csv')
annostat.to_csv('annostat.csv')
