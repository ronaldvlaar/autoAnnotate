import numpy as np
import pandas as pd

a=pd.read_csv('../experiments/l2cs_extendgaze0/33001_sessie1_taskrobotEngagement.csv')


def gazeto3d(gaze):
    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = -np.sin(gaze[1])
    gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt


def angular(gaze, label):
    total = np.sum(gaze * label)
    return np.arccos(min(total/(np.linalg.norm(gaze) * np.linalg.norm(label)), 0.9999999))*180/np.pi


if __name__ == '__main__':
    py = zip(a['yaw'], a['pitch'])
    gazes = []
    angles = []
    for p, y in py:
        gazes.append(gazeto3d([p, y]))
    for i in range(len(gazes)-1):
        angles.append(angular(gazes[i], gazes[i+1]))

print(sum(angles)/len(angles))
