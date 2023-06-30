"""
Author: Ronald Vlaar
Based on: https://github.com/ejcgt/attention-target-detection/blob/master/demo.py

Outputs gaze annotation and classification for a video recording.

File should be placed the main folder of the attention-target-detection project

To run the script:

python demo_model4.py --extendgaze 0 --dest ../experiments/model4_extendgaze0/
"""

import argparse, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
# from scipy.misc import imresize
from matplotlib import cm
# from scipy import misc
from model import ModelSpatial
from utils import imutils, evaluation
from config import *
from readbbtxt import readbbtxt
import cv2
import math
import time

NOFACE = 42

datafolder = '../data_fin/'
datafile = 'pixel_position_vis.txt'

data = readbbtxt(datafolder+datafile)
# remove .png extension from filenames
data['file'] = data['file'].apply(lambda x: x[:-4])


parser = argparse.ArgumentParser()
parser.add_argument('--model_weights', type=str, help='model weights', default='model_demo.pt')
parser.add_argument('--image_dir', type=str, help='images', default='data/demo/frames')
parser.add_argument('--head', type=str, help='head bounding boxes', default='data/demo/person1.txt')
parser.add_argument('--vis_mode', type=str, help='heatmap or arrow', default='arrow')
parser.add_argument('--out_threshold', type=int, help='out-of-frame target dicision threshold', default=100)
parser.add_argument('--dest', dest='dest', help='destination folder', default='../pitchjaw', type=str)
parser.add_argument(
        '--extendgaze', dest='extendgaze', help='Set to 1 if gaze should be extended to a line pointing out of the scene to see if it intersects with a bounding box along the way',
        default=0, type=int)

args = parser.parse_args()


def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)


def inbb(tlx, tly, brx, bry, gazex, gazey):
    return tlx <= gazex <= tlx+(brx-tlx) and tly <= gazey <= tly+(bry-tly)


def get_classname(id):
    if id == 0:
        return 'table'
    elif id == 1:
        return 'robot'
    elif id == 2:
        return 'tablet'
    elif id == 3:
        return 'elsewhere'
    elif id == 4:
        return 'unknown'


def line_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Returns the coordinates px, py. p is the intersection of the lines
    defined by ((x1, y1), (x2, y2)) and ((x3, y3), (x4, y4))
    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    """

    denominator = ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))

    # Lines are parallel
    if denominator == 0:
        return -1, -1
    px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)) / denominator
    py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)) / denominator

    return int(px), int(py)


def extgaze(pos, gazex, gazey, tlx, tly, brx, bry):
    """
    get minimal intersection distance of the gaze line with the bounding boxs specified by tlx, tly, brx, bry.
    Distance is infinity if there is no intersection
    """
    x1, y1 = pos[0], pos[1]
    x2, y2 = gazex, gazey

    # left top right bottom border lines of the bounding box
    lines = [(tlx, 0, tlx, tly), (tlx, tly, brx, bry),
             (brx, 0, brx, bry), (tlx, 0, tly, 0)]

    min_dist = float('inf')
    for x3, y3, x4, y4 in lines:
        xtemp, ytemp = line_intersect(x1, y1, x2, y2, x3, y3, x4, y4)
        if (tlx <= xtemp <= brx) and (tly <= ytemp <= bry):
            # line intersections do not discriminate fordirection of the vector. The boundary with the smallest distance is the true intersection where
            # the gaze vector leaves the image
            dist = math.sqrt((x2-xtemp)**2+(y2-ytemp)**2)
            min_dist = dist if dist < min_dist else min_dist
            if dist == min_dist:
                gazex = xtemp
                gazey = ytemp
                # print(x1, y1, x2, y2, x3, y3, x4, y4)

    return min_dist


def get_gazexy(a, b, c, d, image_in, pitch_pred, jaw_pred):
    (h, w) = image_in.shape[:2]
    length = w/2
    pos = (int(a+c / 2.0), int(b+d / 2.0))
    dx = -length * np.sin(pitch_pred) * np.cos(jaw_pred)
    dy = -length * np.sin(jaw_pred)
    gazex_init, gazey_init = round(pos[0] + dx), round(pos[1] + dy)

    return gazex_init, gazey_init


def classify_gaze(a, b, c, d, w, h, gazex, gazey, filename):
    """
    returns 
    0 : pen & paper
    1 : robot
    2 : tablet
    3 : elsewhere
    4 : unknown
    """

    pos = (int(a+c / 2.0), int(b+d / 2.0))

    gazex_init = gazex
    gazey_init = gazey

    # Check if gaze vector leaves the image
    if gazex < 0 or gazex > w or gazey < 0 or gazey > h:
        # Do line intersection to find at which point the gaze vector line
        # leaves the image.

        # top, right, bottom, left vector line of image boundaries
        x1, y1 = pos[0], pos[1]
        x2, y2 = gazex, gazey
        lines = [(0, 0, w, 0), (w, 0, w, h), (0, h, w, h), (0, 0, 0, h)]

        min_dist = float('inf')
        for x3, y3, x4, y4 in lines:
            gazextemp, gazeytemp = line_intersect(
                x1, y1, x2, y2, x3, y3, x4, y4)
            if (0 <= gazextemp <= w) and (0 <= gazeytemp <= h):
                # line intersections do not discriminate fordirection of the vector. The boundary with the smallest distance is the true intersection where
                # the gaze vector leaves the image
                dist = math.sqrt((x2-gazextemp)**2+(y2-gazeytemp)**2)
                min_dist = dist if dist < min_dist else min_dist
                if dist == min_dist:
                    gazex = gazextemp
                    gazey = gazeytemp
                    # print(x1, y1, x2, y2, x3, y3, x4, y4)

    rec = data[data['file'] == filename].iloc[0]

    # Check small objects first because if e.g. tablet bb intersects with bb of robot and gaze is towards the parts of intersection, changes are higher the child is indeed looking at the
    # smaller bb, thus the tablet.

    return_val = 3
    if inbb(round(rec['tl_tablet_x']*w), round(rec['tl_tablet_y']*h), round(rec['br_tablet_x']*w), round(rec['br_tablet_y']*h), gazex, gazey):
        return 2
    elif inbb(round(rec['tl_pp_x']*w), round(rec['tl_pp_y']*h), round(rec['br_pp_x']*w), round(rec['br_pp_y']*h), gazex, gazey):
        return 0
    elif inbb(round(rec['tl_robot_x']*w), round(rec['tl_robot_y']*h), round(rec['br_robot_x']*w), round(rec['br_robot_y']*h), gazex, gazey):
        return 1

    # No gaze into the objects bb. Extend the gaze to see if there is intersection
    # perform line intersection if args.extendgaze!=0
    if args.extendgaze:
        # intersection distances (infinity if no intersection) with the bounding boxes of the objects
        intersect_distances = [(2, extgaze(pos, gazex_init, gazey_init, round(rec['tl_tablet_x']*w), round(rec['tl_tablet_y']*h), round(rec['br_tablet_x']*w), round(rec['br_tablet_y'])*h)),
                               (0, extgaze(pos, gazex_init, gazey_init, round(rec['tl_pp_x']*w), round(
                                   rec['tl_pp_y']*h), round(rec['br_pp_x']*w), round(rec['br_pp_y'])*h)),
                               (1, extgaze(pos, gazex_init, gazey_init, round(rec['tl_robot_x']*w), round(rec['tl_robot_y']*h), round(rec['br_robot_x']*w), round(rec['br_robot_y'])*h))]

        intersect_distances.sort(key=lambda x: x[1])
        smallest_distance = intersect_distances[0][1]
        object_class = intersect_distances[0][0]
        if smallest_distance < float('inf'):
            return object_class

    # print(w, w, h, get_classname(return_val), gazex, gazey,round(rec['tl_tablet_x']*w), round(rec['tl_tablet_y']*h), round(rec['br_tablet_x']*w), round(rec['br_tablet_y']*h))
    return return_val


def run(f, cap):
    # column_names = ['frame', 'hleft', 'htop', 'hright', 'hbottom']
    df = pd.read_csv('../experiments/l2cs_extendgaze0/'+f+'.csv')
    # df = pd.read_csv(args.head, names=column_names, index_col=0)
    df['hleft'] -= (df['hright']-df['hleft'])*0.1
    df['hright'] += (df['hright']-df['hleft'])*0.1
    df['htop'] -= (df['hbottom']-df['htop'])*0.1
    df['hbottom'] += (df['hbottom']-df['htop'])*0.1

    df['left'] = df['hleft']
    df['right'] = df['hright']
    df['top'] = df['htop']
    df['bottom'] = df['hbottom']

    # set up data transformation
    test_transforms = _get_transform()

    model = ModelSpatial()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.model_weights)
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.cuda()
    model.train(False)

    if not cap.isOpened():
        raise IOError("Could not read the video file")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # output = cv2.VideoWriter(
    #     args.dest+'vids_annotated/'+f +
    #     '.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
    #     fps, (width, height))

    i = -1
    class_ = []
    xs = []
    ys = []
    fps_processed = []
    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                print('All frames are processed')
                break
            i+=1
            start_fps = time.time()
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_raw = Image.fromarray(frame)
            # frame_raw = frame_raw.convert('RGB')
            width, height = frame_raw.size

            head_box = [df.loc[i,'left'], df.loc[i,'top'], df.loc[i,'right'], df.loc[i,'bottom']]


            head = frame_raw.crop((head_box)) # head crop

            head = test_transforms(head) # transform inputs
            frame = test_transforms(frame_raw)
            head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], width, height,
                                                        resolution=input_resolution).unsqueeze(0)

            head = head.unsqueeze(0).cuda()
            frame = frame.unsqueeze(0).cuda()
            head_channel = head_channel.unsqueeze(0).cuda()

            # forward pass
            raw_hm, _, inout = model(frame, head_channel, head)

            # heatmap modulation
            raw_hm = raw_hm.cpu().detach().numpy() * 255
            raw_hm = raw_hm.squeeze()
            inout = inout.cpu().detach().numpy()
            inout = 1 / (1 + np.exp(-inout))
            inout = (1 - inout) * 255
            # norm_map = imresize(raw_hm, (height, width)) - inout
            # norm_map = raw_hm, (height, width) - inout
            imt = Image.fromarray(np.uint8(cm.gist_earth(raw_hm)))
            norm_map = imt.resize((height, width)) - inout

            # vis
            plt.close()
            fig = plt.figure()
            fig.canvas.manager.window.move(0,0)
            plt.axis('off')
            # plt.imshow(frame_raw)

            ax = plt.gca()
            rect = patches.Rectangle((head_box[0], head_box[1]), head_box[2]-head_box[0], head_box[3]-head_box[1], linewidth=2, edgecolor=(0,1,0), facecolor='none')
            ax.add_patch(rect)
    
            myFPS = 1.0 / (time.time() - start_fps)
            fps_processed.append(myFPS)

            if sum(head_box) < 0:
                print(head_box)
                xs.append(-1)
                ys.append(-1)
                class_.append(4)
                continue

            if args.vis_mode == 'arrow':
                if inout < args.out_threshold: # in-frame gaze
                    pred_x, pred_y = evaluation.argmax_pts(raw_hm)
                    norm_p = [pred_x/output_resolution, pred_y/output_resolution]
                    gazex, gazey = (norm_p[0]*width, norm_p[1]*height)
                    xs.append(gazex)
                    ys.append(gazey)

                    # print(gazex, gazey)
                    gaze_class = classify_gaze(df.loc[i,'left'], df.loc[i,'top'], df.loc[i,'right']-df.loc[i,'left'],  df.loc[i,'bottom']-df.loc[i,'top'], width, height, gazex, gazey, f)
                    # print(gaze_class)
                    class_.append(gaze_class)
                    # circ = patches.Circle((norm_p[0]*width, norm_p[1]*height), height/50.0, facecolor=(0,1,0), edgecolor='none')
                    # ax.add_patch(circ)
                    # plt.plot((norm_p[0]*width,(head_box[0]+head_box[2])/2), (norm_p[1]*height,(head_box[1]+head_box[3])/2), '-', color=(0,1,0,1))
                else:
                    xs.append(-1)
                    ys.append(-1)
                    class_.append(3)

                    # pred_x, pred_y = evaluation.argmax_pts(raw_hm)
                    # print(pred_x, pred_y)
                    
            else:
                plt.imshow(norm_map, cmap = 'jet', alpha=0.2, vmin=0, vmax=255)
              

            # plt.tight_layout()
            # plt.savefig(args.dest+'frames_annotated/'+f+str(i)+'.png')
            # plt.show(block=False)
            # plt.pause(0.1)

        dataframe = pd.DataFrame(
            data=np.concatenate(
                [np.array(class_, ndmin=2), np.array(xs, ndmin=2), np.array(ys, ndmin=2), np.array(fps_processed, ndmin=2)]).T,
            columns=['class', 'gazex', 'gazey', 'fps'])
        dataframe.to_csv(args.dest+f+'.csv', index=False)
        # print('DONE!')


if __name__ == "__main__":
    root = '../all_vids/visible_with_bounding_boxes/'
    for f in data['file']:
        video = f+'.MP4'
        print(root+video)
        cap = cv2.VideoCapture(root+video)
        if not cap.isOpened():
            video = f+'.mp4'
            cap = cv2.VideoCapture(root+video)
            if not cap.isOpened():
                raise IOError("Could not read the video file")

        run(f, cap)

    datafolder = '../data_fin/'
    datafile = 'pixel_position_invis_new.txt'

    data = readbbtxt(datafolder+datafile)
    # remove .png extension from filenames
    data['file'] = data['file'].apply(lambda x: x[:-4])

    root = '../all_vids/invisible_with_bounding_boxes/'
    for f in data['file']:
        video = f+'.MP4'
        print(root+video)
        cap = cv2.VideoCapture(root+video)
        if not cap.isOpened():
            video = f+'.mp4'
            cap = cv2.VideoCapture(root+video)
            if not cap.isOpened():
                raise IOError("Could not read the video file")

        run(f, cap)
