import pandas as pd
import numpy as np
from readbbtxt_inv import readbbtxt

import cv2
from PIL import Image

datafolder = './data_fin/'
datafile = 'pixel_position_invis_new.txt'

data = readbbtxt(datafolder + datafile)

# def add_margin(pil_img, top, right, bottom, left, color):
#     width, height = pil_img.size
#     new_width = width + right + left
#     new_height = height + top + bottom
#     result = Image.new(pil_img.mode, (new_width, new_height), color)
#     result.paste(pil_img, (left, top))
#     return result


root = './invisible/'
for f in data['file']:
    video = f[:-3] + 'MP4'
    cap = cv2.VideoCapture(root + video)
    if not cap.isOpened():
        video = f[:-3] + 'mp4'
        cap = cv2.VideoCapture(root + video)
        if not cap.isOpened():
            raise IOError("Could not read the video file")

        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps = int(cap.get(cv2.CAP_PROP_FPS))
        # cap = Image.open(cap)
        # if cap.size != (852, 480):
        #     cap = cap.resize((852, 480), Image.LANCZOS)
        #     # width, height = frame.size
        #     cap = add_margin(cap, 0, round(width / 8), round(height * 0.75), round(width / 8), (0, 0, 0))
        # # width, height = cap.size
        # cap = add_margin(cap, 0, round(width / 8), round(height * 0.75), round(width / 8), (0, 0, 0))

    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    old_width = 852
    old_height = 480
    width = round(old_width * 1.25)
    height = round(old_height * 1.75)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # cap = Image.open(cap)
    # if cap.size != (852, 480):
    #     cap = cap.resize((852, 480), Image.LANCZOS)
    #     # width, height = frame.size
    #     cap = add_margin(cap, 0, round(width / 8), round(height * 0.75), round(width / 8), (0, 0, 0))
    # # width, height = cap.size
    # cap = add_margin(cap, 0, round(width / 8), round(height * 0.75), round(width / 8), (0, 0, 0))
    # output = cv2.VideoWriter(
    #     './invisible_bb/' + video, cv2.VideoWriter_fourcc(*'MPEG'),
    #     fps, (width, height))
    output = cv2.VideoWriter(
        './invisible_bb/' + video, cv2.VideoWriter_fourcc('m','p','4','v'),
        fps, (width, height))

    while (True):
        ret, frame = cap.read()
        # read image
        # cv2.imwrite(frame_path + "%05d.jpg" % count, image)
        # frame = cv2.imread(frame)
        # frame = cv2.resize(frame, (852, 480))
        # height_h, width_w, channels = frame.shape
        # frame = cv2.copyMakeBorder(frame, 0, round(width_w / 8), round(height_h * 0.75), round(width_w / 8),
        #                            cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # frame = Image.open(frame)
        # if frame.size != (852, 480):
        #     frame = frame.resize((852, 480), Image.LANCZOS)
        #     # width, height = frame.size
        #     frame = add_margin(frame, 0, round(width / 8), round(height * 0.75), round(width / 8), (0, 0, 0))
        # # width, height = cap.size
        # frame = add_margin(frame, 0, round(width / 8), round(height * 0.75), round(width / 8), (0, 0, 0))

        if (ret):
            # adding bb on each frame
            d = data[data['file'] == f].iloc[0]

            # frame = cv2.imread(frame)
            frame = cv2.resize(frame, (852, 480))
            height_h, width_w, channels = frame.shape
            # print(height_h)
            frame = cv2.copyMakeBorder(frame, 0, round(height_h * 0.75), round(width_w / 8), round(width_w / 8),
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))

            # Red rectangle (tablet)
            cv2.rectangle(frame, (int(d['tl_tablet_x'] * width), int(d['tl_tablet_y'] * height)),
                          (int(d['br_tablet_x'] * width), int(d['br_tablet_y'] * height)), (0, 0, 255), 0)
            cv2.putText(frame, 'Tablet', (int(d['tl_tablet_x'] * width), int(d['tl_tablet_y'] * height) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # Green rectangle (robot)
            cv2.rectangle(frame, (int(d['tl_robot_x'] * width), int(d['tl_robot_y'] * height)),
                          (int(d['br_robot_x'] * width), int(d['br_robot_y'] * height)), (0, 255, 0), 0)
            cv2.putText(frame, 'Robot', (int(d['tl_robot_x'] * width), int(d['tl_robot_y'] * height) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Blue rectangle (table)
            cv2.rectangle(frame, (int(d['tl_pp_x'] * width), int(d['tl_pp_y'] * height)),
                          (int(d['br_pp_x'] * width), int(d['br_pp_y'] * height)), (255, 0, 0), 0)
            cv2.putText(frame, 'Table', (int(d['tl_pp_x'] * width), int(d['tl_pp_y'] * height) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # # Red rectangle (tablet)
            # cv2.rectangle(frame, (int(d['tl_tablet_x'][0]* width), int(d['tl_tablet_y'][0] * height)),
            #               (int(d['br_tablet_x'][0] * width), int(d['br_tablet_y'][0] * height)), (0, 0, 255), 0)
            # cv2.putText(frame, 'Tablet', (int(d['tl_tablet_x'][0] * width), int(d['tl_tablet_y'][0] * height) - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # # Green rectangle (robot)
            # cv2.rectangle(frame, (int(d['tl_robot_x'][0] * width), int(d['tl_robot_y'][0] * height)),
            #               (int(d['br_robot_x'][0] * width), int(d['br_robot_y'][0] * height)), (0, 255, 0), 0)
            # cv2.putText(frame, 'Robot', (int(d['tl_robot_x'][0] * width), int(d['tl_robot_y'][0] * height) - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # # Blue rectangle (table)
            # cv2.rectangle(frame, (int(d['tl_pp_x'][0] * width), int(d['tl_pp_y'][0] * height)),
            #               (int(d['br_pp_x'][0] * width), int(d['br_pp_y'][0] * height)), (255, 0, 0), 0)
            # cv2.putText(frame, 'Table', (int(d['tl_pp_x'][0] * width), int(d['tl_pp_y'][0] * height) - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # original coordinates
            # Red rectangle (tablet)
            # cv2.rectangle(frame, (int(d['tl_tablet_x']), int(d['tl_tablet_y'])), (int(d['br_tablet_x']), int(d['br_tablet_y'])),
            #               (0, 0, 255), 0)
            # # Green rectangle (robot)
            # cv2.rectangle(frame, (int(d['tl_robot_x']), int(d['tl_robot_y'])), (int(d['br_robot_x']), int(d['br_robot_y'])),
            #               (0, 255, 0), 0)
            # # Blue rectangle (table)
            # cv2.rectangle(frame, (int(d['tl_pp_x']), int(d['tl_pp_y'])), (int(d['br_pp_x']), int(d['br_pp_y'])),
            #               (255, 0, 0), 0)

            # writing the new frame in output
            output.write(frame)
            cv2.imshow(f, frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
        else:
            break

    cv2.destroyAllWindows()
    output.release()
    cap.release()

    # comment break to output all videos
    # break
