
import pandas as pd
import math
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy import stats


def aggregateivt(threshold=20, min_fixation=2, dest='./aggr_l2csextendgaze0/',
                 src='../experiments/l2cs_extendgaze0/', files=None):
    """
    Aggregation algorithm based on:
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8987791
    # https://dl.acm.org/doi/pdf/10.1145/355017.355028

    It detects fixation based on the angle change between two frames
    if the angular difference is higher than the threshold, this means there is a saccade
    saccades are converted to class 4. If there is variance in terms of the classes for
    a fixations group the mode is chosen
    """
    # file = '33001_sessie1_taskrobotEngagement'
    not_aggregated_ck = []
    aggregated_ck = []
    not_aggregated_ckw4 = []
    aggregated_ckw4 = []
    annotations = pd.read_csv('../manual_annotation/annotations_frame.csv')
    vidstats = pd.read_csv('vidstat.csv')
    files = annotations['file'].unique() if files is None else files
    for file in files:
        ann = annotations[annotations['file'] == file]['class']
        fps = int(vidstats[vidstats['file'] == file]['rate'].iloc[0])
        tot_frames = vidstats[vidstats['file'] == file]['frames'].iloc[0]
        dat = pd.read_csv(src+file+'.csv')
        dat = dat.reset_index()
        dat['class'] = dat['class'].astype(int)
        angles = []

        for idx, row in dat.iterrows():
            tlx = row['hleft']
            tly = row['htop']
            brx = row['hright']
            bry = row['hbottom']

            eyex, eyey = (int((tlx+brx) / 2.0), int((tly+bry) / 2.0))
            slope1 = abs(row['gazey']-eyey)/abs(row['gazex']-eyex)

            if idx+1 < len(dat):
                nextrow = dat.loc[idx+1]

                tlx = nextrow['hleft']
                tly = nextrow['htop']
                brx = nextrow['hright']
                bry = nextrow['hbottom']

                eyex, eyey = (int((tlx+brx) / 2.0), int((tly+bry) / 2.0))
                slope2 = abs(nextrow['gazey']-eyey)/abs(nextrow['gazex']-eyex)
                angle = math.degrees(
                    math.atan(abs((slope1-slope2)/(1+slope1*slope2))))
                angles.append(angle)

                # if math.isnan(angle):
                #     print(angle, 'a')

        angles = np.array(angles)
        print('angles', np.nanmean(angles), angles)
        fixation_groups = list(np.where(angles > threshold)[0])
        fixation_groups = np.append(fixation_groups, tot_frames)
        # print(fixation_groups)
        class_aggregated = []
        last = 0
        # for fixation in fixation_groups:
        #     if last == fixation:
        #         continue
        #     mode = 4
        #     if len(range(last,fixation)) >=min_fixation*fps:
        #         mode = stats.mode(dat['class'][last:fixation], keepdims=True)[0][0]
        #         # mode = dat['class'].iloc[last+13]

        #     for _ in range(last, fixation):
        #         class_aggregated.append(mode)
        #     # print(stats.mode(dat['class'][last:fixation])[0][0])
        #     last=fixation

        for fixation in fixation_groups:
            if last == fixation:
                continue
            if len(range(last, fixation)) <= int(min_fixation*fps):
                mode = 4
                for _ in range(last, fixation):
                    class_aggregated.append(mode)
            elif len(range(last, fixation)) >= 2*fps:
                mode = stats.mode(
                    dat['class'][last:fixation], keepdims=True)[0][0]
                # w3 = list(filter(lambda a: a != 3, dat['class'][last:fixation]))
                # mode = stats.mode(w3, keepdims=True)[0][0]
                for _ in range(last, fixation):
                    class_aggregated.append(mode)
            else:
                for i in range(last, fixation):
                    class_aggregated.append(dat['class'].iloc[i])

            # print(stats.mode(dat['class'][last:fixation])[0][0])
            last = fixation

        ck = cohen_kappa_score(dat['class'].astype(
            int), ann, labels=[0, 1, 2, 3, 4])
        ckw4 = cohen_kappa_score(
            dat['class'].astype(int), ann, labels=[0, 1, 2, 3])
        print(ck, ckw4)
        not_aggregated_ck.append(ck)
        not_aggregated_ckw4.append(ckw4)

        ck = cohen_kappa_score(class_aggregated, ann, labels=[0, 1, 2, 3, 4])
        ckw4 = cohen_kappa_score(class_aggregated, ann, labels=[0, 1, 2, 3])
        print(ck, ckw4)
        aggregated_ck.append(ck)
        aggregated_ckw4.append(ckw4)

        dat['class'] = class_aggregated

        dat.to_csv(dest+file+'.csv')
        # annotations[annotations['file'] == file].to_csv(file+'_mann.csv')

    # print('a',angles > threshold)


def get_max_dispersion(points, w, h):
    xmin, xmax, ymin, ymax = float('inf'), 0, float('inf'), 0
    for x, y in points:
        xmin = x if x < xmin else xmin
        ymin = y if y < ymin else ymin
        xmax = x if x > xmax else xmax
        ymax = y if y > ymax else ymax

    return (abs(xmax-xmin)/w + abs(ymax-ymin)/h)/2


def aggregatedti(threshold=2, min_fixation=2, dest='./aggr_l2csextendgaze0/',
                 src='../experiments/l2cs_extendgaze0/', files=None, thresholds=None, getstats=True):
    """
    https://link.springer.com/content/pdf/10.3758/APP.71.4.881.pdf
    """
    # file = '33001_sessie1_taskrobotEngagement'
    annotations = pd.read_csv('../manual_annotation/annotations_frame.csv')
    vidstats = pd.read_csv('vidstat.csv')
    files = annotations['file'].unique() if files is None else files
    not_aggregated_ck = []
    aggregated_ck = []
    not_aggregated_ckw4 = []
    aggregated_ckw4 = []
    dispersions_l = []
    dispersions_avg = []

    for idx, file in enumerate(files):
        if thresholds is not None:
            threshold = thresholds[idx]
        ann = annotations[annotations['file'] == file]['class']
        fps = int(vidstats[vidstats['file'] == file]['rate'].iloc[0])
        # tot_frames = vidstats[vidstats['file'] == file]['frames'].iloc[0]
        w = vidstats[vidstats['file'] == file]['width'].iloc[0]
        h = vidstats[vidstats['file'] == file]['height'].iloc[0]
        dat = pd.read_csv(src+file+'.csv')
        dat = dat.reset_index()
        dat['class'] = dat['class'].astype(int)

        pointst = list(zip(dat['gazex'], dat['gazey']))

        # maxgazex = int(dat['gazex'].abs().max())
        # maxgazey = int(dat['gazey'].abs().max())

        # w = maxgazex
        # h = maxgazey

        points = []
        dispersions = []
        for x, y in pointst:
            if x < 0:
                x = 0
            if x > w:
                x = w
            if y < 0:
                y = 0
            if y > h:
                y = h
            points.append((x, y))
        # window = [points.pop(), points.pop()]
        # last = 0

        window = []
        windowleft = 0
        windowright = 0
        fixations = []

        while len(points) > 0:
            window.append(points.pop(0))
            windowright += 1
            # fill initial window
            while len(window) < int(min_fixation*fps) and len(points) > 0:
                window.append(points.pop(0))
                windowright += 1
            # check dispersion
            disp = get_max_dispersion(window, w, h)
            dispersions.append(disp)
            # print(len(window), disp, threshold)
            if disp > threshold and len(window) == int(min_fixation*fps):
                windowleft += 1
                windowright = windowleft
                for _ in range(len(window)-1):
                    points.insert(0, window[::-1].pop())
                window = []
                continue
            elif disp > threshold and len(window) > int(min_fixation*fps):
                fixations.append((windowleft, windowright))
                windowleft = windowright
                window = []

        class_aggregated = []
        last_end = 0
        # print(fixations, max(dispersions), min(dispersions),
        #       sum(dispersions)/len(dispersions))

        for id, (start, end) in enumerate(fixations):
            for i in range(last_end, start):
                mode = dat['class'].loc[i]
                # mode = 4 if (start-last_end) < fps else mode
                # mode = 4
                # if (start-last_end) < min_fixation*fps else stats.mode(
                #     dat['class'][last_end:start], keepdims=True)[0][0]
                # mode = stats.mode(dat['class'][last_end:start], keepdims=True)[0][0] if (start-last_end) > 5*int(min_fixation*fps) else mode
                class_aggregated.append(mode)

            mode = stats.mode(dat['class'][start:end], keepdims=True)[0][0]
            for _ in range(start, end):
                class_aggregated.append(mode)

            if (id+1) == len(fixations):
                for i in range(end, len(dat['class'])):
                    mode = dat['class'].loc[i]
                    # mode = 4 if (len(dat['class'])-end) < fps else mode
                    # mode = 4
                    # if (len(dat['class'])-end) < min_fixation*fps else stats.mode(
                    #     dat['class'][end:len(dat['class'])], keepdims=True)[0][0]
                    # mode = stats.mode(dat['class'][end:len(dat['class'])], keepdims=True)[0][0] if (start-last_end) > 5*int(min_fixation*fps) else mode
                    class_aggregated.append(mode)

            last_end = end
        if len(fixations) == 0:
            class_aggregated = list(dat['class'])
        # print(class_aggregated, ann, set(class_aggregated), set(ann))

        if getstats:
            ck = cohen_kappa_score(dat['class'].astype(
                int), ann, labels=[0, 1, 2, 3, 4])
            ckw4 = cohen_kappa_score(
                dat['class'].astype(int), ann, labels=[0, 1, 2, 3])
            # print(ck, ckw4)
            not_aggregated_ck.append(ck)
            not_aggregated_ckw4.append(ckw4)

            ck = cohen_kappa_score(class_aggregated, ann,
                                   labels=[0, 1, 2, 3, 4])
            ckw4 = cohen_kappa_score(
                class_aggregated, ann, labels=[0, 1, 2, 3])
            # print(ck, ckw4)
            # print()
            aggregated_ck.append(ck)
            aggregated_ckw4.append(ckw4)

        dispersions_l.append(dispersions)
        dispersions_avg.append(sum(dispersions)/len(dispersions))

        dat['class'] = class_aggregated

        dat.to_csv(dest+file+'.csv')

    return np.array(aggregated_ck), np.array(not_aggregated_ck), np.array(aggregated_ckw4), np.array(not_aggregated_ckw4), dispersions_avg


def smooth(window_len=24, dest='./aggr_l2csextendgaze0/',
           src='../experiments/l2cs_extendgaze0/', files=None, getstats=True):
    annotations = pd.read_csv('../manual_annotation/annotations_frame.csv')
    vidstats = pd.read_csv('vidstat.csv')
    files = annotations['file'].unique() if files is None else files
    not_aggregated_ck = []
    aggregated_ck = []
    not_aggregated_ckw4 = []
    aggregated_ckw4 = []

    for _, file in enumerate(files):

        ann = annotations[annotations['file'] == file]['class']
        fps = int(vidstats[vidstats['file'] == file]['rate'].iloc[0])
        # tot_frames = vidstats[vidstats['file'] == file]['frames'].iloc[0]
        w = vidstats[vidstats['file'] == file]['width'].iloc[0]
        h = vidstats[vidstats['file'] == file]['height'].iloc[0]
        dat = pd.read_csv(src+file+'.csv')
        dat = dat.reset_index()
        dat['class'] = dat['class'].astype(int)
        datt=list(dat['class'])
        # window_len = int(fps)
        most_freq_val = lambda x: stats.mode(x, keepdims=True)[0][0]
        # smoothed = [most_freq_val(datt[i:i+window_len]) for i in range(0,len(datt)-window_len+1)]
        # [smoothed.insert(0, most_freq_val(datt[0:window_len])) for _ in range(window_len-1)]
        smoothed = [most_freq_val(datt[i:i+window_len]) for i in range(len(datt)-window_len+1)]
        # print(len(datt), len(smoothed), len(datt)-len(smoothed))
        # [smoothed.append(most_freq_val(datt[len(datt)-window_len:len(datt)])) for _ in range(len(datt)-len(smoothed))]

        [smoothed.append(smoothed[-1]) for _ in range(window_len-1-int((window_len-1)/2))]
        [smoothed.insert(0, smoothed[0]) for _ in range(int((window_len-1)/2))]
        
            
        class_aggregated = smoothed
        if getstats:
            ck = cohen_kappa_score(dat['class'].astype(
                int), ann, labels=[0, 1, 2, 3, 4])
            ckw4 = cohen_kappa_score(
                dat['class'].astype(int), ann, labels=[0, 1, 2, 3])
            # print(ck, ckw4)
            not_aggregated_ck.append(ck)
            not_aggregated_ckw4.append(ckw4)

            ck = cohen_kappa_score(class_aggregated, ann,
                                   labels=[0, 1, 2, 3, 4])
            ckw4 = cohen_kappa_score(
                class_aggregated, ann, labels=[0, 1, 2, 3])
            # print(ck, ckw4)
            # print()
            aggregated_ck.append(ck)
            aggregated_ckw4.append(ckw4)

        # print(class_aggregated)
        dat['class'] = class_aggregated
        dat.to_csv(dest+file+'.csv')

    return np.array(aggregated_ck), np.array(not_aggregated_ck), np.array(aggregated_ckw4), np.array(not_aggregated_ckw4)


if __name__ == '__main__':
      # aggregateivt(threshold=20, min_fixation=0.5, dest='./aggr/',
    #              src='', files=['33001_sessie1_taskrobotEngagement'])

    # aggregatedti(threshold=1/5, min_fixation=0.2, dest='./aggr/',
    #              src='', files=['33001_sessie1_taskrobotEngagement'])

    # first learn average dispersions between every two frames


    # smoothing method combined. First aggregation, then smoothing
    # maxs0 = 24
    # maxs1 = 24
    # max_aggext0 = 0
    # max_aggext1 = 0


    # for s in range(3, 24*6, 3):
    #     print('smoothing', s)
    #     aggregated_ck, not_aggregated_ck,  aggregated_ckw4, not_aggregated_ckw4 = smooth(window_len=s, dest='./aggr_smooth0/', src='../experiments/l2cs_extendgaze0/', getstats=True)
    #     print(not_aggregated_ck.mean(), not_aggregated_ckw4.mean())
    #     print(aggregated_ck.mean(), aggregated_ckw4.mean())
    #     print()

    #     if aggregated_ckw4.mean() > max_aggext0:
    #         max_aggext0 = aggregated_ckw4.mean()
    #         maxs0 = s

    #     aggregated_ck, not_aggregated_ck,  aggregated_ckw4, not_aggregated_ckw4 = smooth(window_len=s, dest='./aggr_smooth1/', src='../experiments/l2cs_extendgaze1/', getstats=True)
    #     print(not_aggregated_ck.mean(), not_aggregated_ckw4.mean())
    #     print(aggregated_ck.mean(), aggregated_ckw4.mean())
    #     print()

    #     if aggregated_ckw4.mean() > max_aggext1:
    #         max_aggext1 = aggregated_ckw4.mean()
    #         maxs1 = s
            

    # print(maxs0, maxs1, max_aggext0, max_aggext1)

    


    print('dti')
    aggregated_ck, not_aggregated_ck,  aggregated_ckw4, not_aggregated_ckw4, dispersions_avg = aggregatedti(threshold=0, min_fixation=0.1, dest='./aggr_l2csextendgaze0/',
                                                                                                            src='../experiments/l2cs_extendgaze0/', getstats=False)

    aggregated_ck, not_aggregated_ck,  aggregated_ckw4, not_aggregated_ckw4, _ = aggregatedti(threshold=1/5, min_fixation=0.1, dest='./aggr_l2csextendgaze0/',
                                                                                              src='../experiments/l2cs_extendgaze0/', thresholds=dispersions_avg, getstats=True)
    print(not_aggregated_ck.mean(), not_aggregated_ckw4.mean())
    print(aggregated_ck.mean(), aggregated_ckw4.mean())
    print()

    # first learn average dispersions between every two frames
    aggregated_ck, not_aggregated_ck,  aggregated_ckw4, not_aggregated_ckw4, dispersions_avg = aggregatedti(threshold=0, min_fixation=0.1, dest='./aggr_l2csextendgaze1/',
                                                                                                            src='../experiments/l2cs_extendgaze1/', getstats=False)

    aggregated_ck, not_aggregated_ck,  aggregated_ckw4, not_aggregated_ckw4, _ = aggregatedti(threshold=1/5, min_fixation=0.1, dest='./aggr_l2csextendgaze1/',
                                                                                              src='../experiments/l2cs_extendgaze1/', thresholds=dispersions_avg, getstats=True)
    print(not_aggregated_ck.mean(), not_aggregated_ckw4.mean())
    print(aggregated_ck.mean(), aggregated_ckw4.mean())
    print()


    # smoothing method
    print('smoothing')
    aggregated_ck, not_aggregated_ck,  aggregated_ckw4, not_aggregated_ckw4 = smooth(window_len=24, dest='./aggr_smooth0/', src='../experiments/l2cs_extendgaze0/', getstats=True)
    print(not_aggregated_ck.mean(), not_aggregated_ckw4.mean())
    print(aggregated_ck.mean(), aggregated_ckw4.mean())
    print()

    aggregated_ck, not_aggregated_ck,  aggregated_ckw4, not_aggregated_ckw4 = smooth(window_len=24, dest='./aggr_smooth1/', src='../experiments/l2cs_extendgaze1/', getstats=True)
    print(not_aggregated_ck.mean(), not_aggregated_ckw4.mean())
    print(aggregated_ck.mean(), aggregated_ckw4.mean())
    print()

    # smoothing method combined. First aggregation, then smoothing
    print('aggregation-smoothing')
    aggregated_ck, not_aggregated_ck,  aggregated_ckw4, not_aggregated_ckw4 = smooth(window_len=24, dest='./aggr_l2csextendgaze0smoothed/', src='./aggr_l2csextendgaze0/', getstats=True)
    print(not_aggregated_ck.mean(), not_aggregated_ckw4.mean())
    print(aggregated_ck.mean(), aggregated_ckw4.mean())
    print()

    aggregated_ck, not_aggregated_ck,  aggregated_ckw4, not_aggregated_ckw4 = smooth(window_len=24, dest='./aggr_l2csextendgaze1smoothed/', src='./aggr_l2csextendgaze1/', getstats=True)
    print(not_aggregated_ck.mean(), not_aggregated_ckw4.mean())
    print(aggregated_ck.mean(), aggregated_ckw4.mean())

    print()

    # first smoothing then aggregation
    print('smoothing-aggregation')
    aggregated_ck, not_aggregated_ck,  aggregated_ckw4, not_aggregated_ckw4, dispersions_avg = aggregatedti(threshold=0, min_fixation=0.1, dest='./aggr_smoothedl2csextendgaze0/',
                                                                                                            src='./aggr_smooth0/', getstats=False)

    aggregated_ck, not_aggregated_ck,  aggregated_ckw4, not_aggregated_ckw4, _ = aggregatedti(threshold=1/5, min_fixation=0.1, dest='./aggr_smoothedl2csextendgaze0/',
                                                                                              src='./aggr_smooth0/', thresholds=dispersions_avg, getstats=True)
    print(not_aggregated_ck.mean(), not_aggregated_ckw4.mean())
    print(aggregated_ck.mean(), aggregated_ckw4.mean())
    print()

    aggregated_ck, not_aggregated_ck,  aggregated_ckw4, not_aggregated_ckw4, dispersions_avg = aggregatedti(threshold=0, min_fixation=0.1, dest='./aggr_smoothedl2csextendgaze1/',
                                                                                                            src='./aggr_smooth1/', getstats=False)

    aggregated_ck, not_aggregated_ck,  aggregated_ckw4, not_aggregated_ckw4, _ = aggregatedti(threshold=1/5, min_fixation=0.1, dest='./aggr_smoothedl2csextendgaze1/',
                                                                                              src='./aggr_smooth1/', thresholds=dispersions_avg, getstats=True)
    print(not_aggregated_ck.mean(), not_aggregated_ckw4.mean())
    print(aggregated_ck.mean(), aggregated_ckw4.mean())
    print()

