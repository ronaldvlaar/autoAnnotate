"""
Author: Ronald Vlaar
Based on: https://github.com/Ahmednull/L2CS-Net

Outputs gaze annotation and classification for a video recording and saves it to the pitchjaw folder

File should be placed the L2CS main folder which may be downloaded from https://github.com/Ahmednull/L2CS-Net

To run the script:

python demo_annotate.py \
 --snapshot models/L2CSNet_gaze360.pkl \
 --gpu 0
 --extendgaze 1
"""

import argparse
import numpy as np
import cv2
import time
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from utils import select_device, draw_gaze
from PIL import Image, ImageOps

from face_detection import RetinaFace
from model import L2CS

import pandas as pd
import math
from readbbtxt import readbbtxt

import cv2

NOFACE = 42

datafolder = '../data_fin/'
datafile = 'pixel_position_vis.txt'

data = readbbtxt(datafolder+datafile)
# remove .png extension from filenames
data['file'] = data['file'].apply(lambda x: x[:-4])


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--cam', dest='cam_id', help='Camera device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)
    parser.add_argument(
        '--extendgaze', dest='extendgaze', help='Set to 1 if gaze should be extended to a line pointing out of the scene to see if it intersects with a bounding box along the way',
        default=1, type=int)

    args = parser.parse_args()
    return args


def getArch(arch, bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                  'The default value of ResNet50 will be used instead!')
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model


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


def classify_gaze(a, b, c, d, image_in, pitch_pred, jaw_pred, filename, args):
    """
    returns 
    0 : pen & paper
    1 : robot
    2 : tablet
    3 : elsewhere
    4 : unknown
    """

    if pitch_pred == NOFACE and jaw_pred == NOFACE:
        return 4

    (h, w) = image_in.shape[:2]
    length = w/2
    pos = (int(a+c / 2.0), int(b+d / 2.0))
    dx = -length * np.sin(pitch_pred) * np.cos(jaw_pred)
    dy = -length * np.sin(jaw_pred)
    gazex_init, gazey_init = round(pos[0] + dx), round(pos[1] + dy)
    gazex, gazey = gazex_init, gazey_init

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


def get_largest_face(faces):
    """
    Returns the face closest to the camera (based on size of the face relative to the image size)
    """
    largest_face_idx = 0
    largest_face = 0
    for idx, face in enumerate(faces):
        box, _, _ = face
        x_min = int(box[0])
        if x_min < 0:
            x_min = 0
        y_min = int(box[1])
        if y_min < 0:
            y_min = 0
        x_max = int(box[2])
        y_max = int(box[3])
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        fsize = bbox_width*bbox_height
        if fsize > largest_face:
            largest_face = fsize
            largest_face_idx = idx

    return faces[largest_face_idx]


def annotate(model, softmax, detector, idx_tensor, cap, filename, args):
    # Check if the file is opened correctly
    if not cap.isOpened():
        raise IOError("Could not read the video file")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output = cv2.VideoWriter(
        '../pitchjaw/vids_annotated/'+filename +
        '.mp4', cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (width, height))

    pitch_predicted_ = []
    yaw_predicted_ = []
    gaze_class_ = []
    fps_processed = []

    headboxl = []
    headboxt = []
    headboxr = []
    headboxb = []

    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                print('All frames are processed')
                break
            start_fps = time.time()

            faces = detector(frame)

            faces = [face for face in faces if face[2] >= 0.95]

            if len(faces) > 0:
                # Assume biggest face in scene is the childs face. This is the only face relevant
                # for the gaze estimation
                box, landmarks, score = get_largest_face(faces)
                x_min = int(box[0])
                if x_min < 0:
                    x_min = 0
                y_min = int(box[1])
                if y_min < 0:
                    y_min = 0
                x_max = int(box[2])
                y_max = int(box[3])
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min

                # Crop image
                img = frame[y_min:y_max, x_min:x_max]
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                img = transformations(im_pil)
                img = Variable(img).cuda(gpu)
                img = img.unsqueeze(0)

                # gaze prediction
                gaze_pitch, gaze_yaw = model(img)

                pitch_predicted = softmax(gaze_pitch)
                yaw_predicted = softmax(gaze_yaw)

                # Get continuous predictions in degrees.
                pitch_predicted = torch.sum(
                    pitch_predicted.data[0] * idx_tensor) * 4 - 180
                yaw_predicted = torch.sum(
                    yaw_predicted.data[0] * idx_tensor) * 4 - 180

                pitch_predicted = pitch_predicted.cpu().detach().numpy() * np.pi/180.0
                yaw_predicted = yaw_predicted.cpu().detach().numpy() * np.pi/180.0

                pitch_predicted_.append(pitch_predicted)
                yaw_predicted_.append(yaw_predicted)

                out_frame = draw_gaze(x_min, y_min, bbox_width, bbox_height, frame,
                                      (pitch_predicted, yaw_predicted), color=(0, 0, 255))
                cv2.rectangle(frame, (x_min, y_min),
                              (x_max, y_max), (0, 255, 0), 1)
                gaze_class = classify_gaze(
                    x_min, y_min, bbox_width, bbox_height, frame, pitch_predicted_[-1], yaw_predicted_[-1], filename, args)
                headboxl.append(box[0])
                headboxt.append(box[1])
                headboxr.append(box[2])
                headboxb.append(box[3])

            else:
                out_frame = frame
                # No face detected, pitch, jaw annotated with 42, 42 to specify that
                pitch_predicted_.append(NOFACE)
                yaw_predicted_.append(NOFACE)
                gaze_class = classify_gaze(
                    0, 0, 0, 0, frame, pitch_predicted_[-1], yaw_predicted_[-1], filename, args)
                headboxl.append(-1)
                headboxt.append(-1)
                headboxr.append(-1)
                headboxb.append(-1)

            gaze_class_.append(gaze_class)
            # store img

            myFPS = 1.0 / (time.time() - start_fps)
            fps_processed.append(myFPS)
            # cv2.putText(frame, 'FPS: {:.1f}'.format(
            #     myFPS), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, '{} (class {})'.format(
                get_classname(gaze_class), gaze_class), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

            output.write(out_frame)
            # cv2.imshow("Demo", frame)
            # if cv2.waitKey(1) & 0xFF == 27:
            #     break

            # from time import sleep
            # sleep(0.5)

        dataframe = pd.DataFrame(
            data=np.concatenate(
                [np.array(pitch_predicted_, ndmin=2), np.array(yaw_predicted_, ndmin=2), np.array(gaze_class_, ndmin=2), np.array(fps_processed, ndmin=2),
                 np.array(headboxl, ndmin=2),np.array(headboxt, ndmin=2),np.array(headboxr, ndmin=2),np.array(headboxb, ndmin=2)]).T,
            columns=['yaw', 'pitch', 'class', 'fps', 'hleft', 'htop', 'hright', 'hbottom'])
        dataframe.to_csv('../pitchjaw/'+filename+'.csv', index=False)


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch = args.arch
    batch_size = 1
    # cam = args.cam_id
    gpu = select_device(args.gpu_id, batch_size=batch_size)
    snapshot_path = args.snapshot

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    model = getArch(arch, 90)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    model.eval()

    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    x = 0

    # Call annotate function
    # filename = '33001_sessie3_taskrobotEngagement'
    # ext = '.MP4'
    # file = filename+ext
    # cap = cv2.VideoCapture('../all_vids/visible_with_bounding_boxes/'+file)
    # annotate(model, softmax, detector, idx_tensor, cap, filename, args)

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

        annotate(model, softmax, detector, idx_tensor, cap, f, args)

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

        annotate(model, softmax, detector, idx_tensor, cap, f, args)
