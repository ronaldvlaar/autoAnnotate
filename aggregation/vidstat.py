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

def get_stats(root, data, cols=['file', 'frames', 'rate', 'width', 'height']):
    vidstats = pd.DataFrame(columns=cols)
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
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
        vidstats.loc[len(vidstats)] = [f, length, fps, width, height]
       
    return vidstats


vidstatsvis = get_stats('../all_vids/visible_with_bounding_boxes/', datavis)
vidstatsinv = get_stats(
    '../all_vids/invisible_with_bounding_boxes/', datainvis)
vidstat = pd.concat([vidstatsvis, vidstatsinv], axis=0)
vidstat.to_csv('vidstat.csv')