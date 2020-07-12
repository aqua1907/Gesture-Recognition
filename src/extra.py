from scipy.spatial import distance as dist
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import cv2

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2

np.set_printoptions(precision=4, suppress=True)

classes = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'K',
    10: 'L',
    11: 'M',
    12: 'N',
    13: 'O',
    14: 'P',
    15: 'Q',
    16: 'R',
    17: 'S',
    18: 'T',
    19: 'U',
    20: 'V',
    21: 'W',
    22: 'X',
    23: 'Y'
}

#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]


def append_points(data, joints):
    """
    Saving coordinate (x, y) of each joint to the list. Overall it's 21 coordinates.
    :param data: list for saving
    :param joints: predicted joints by the regressor model (21, 2) array

    :return: appending (21, 2) array to the data list
    """
    data.append(joints.tolist())
    print(len(data))


def calc_distances(joints):
    """
    Calculate euclidean distances between given two pairs of 2-D coordinates
    :param joints: predicted joints by regressor model (21, 2) array
    :return: list with 9 distances
    """
    dist_20_0 = dist.euclidean(joints[20], joints[0])
    dist_16_0 = dist.euclidean(joints[16], joints[0])
    dist_12_0 = dist.euclidean(joints[12], joints[0])
    dist_8_0 = dist.euclidean(joints[8], joints[0])
    dist_4_0 = dist.euclidean(joints[4], joints[0])
    dist_20_16 = dist.euclidean(joints[20], joints[16])
    dist_16_12 = dist.euclidean(joints[16], joints[12])
    dist_12_8 = dist.euclidean(joints[12], joints[8])
    dist_8_4 = dist.euclidean(joints[8], joints[4])

    return [dist_20_0, dist_16_0, dist_12_0, dist_8_0, dist_4_0,
            dist_20_16, dist_16_12, dist_12_8, dist_8_4]


def create_csv(data, output, columns):
    """
    Write DataFrame to csv file
    :param data: data to write to CSV
    :param output: output path
    :param columns: columns to show in CSV
    :return: created CSV file
    """
    df = pd.DataFrame(data,
                      columns=columns)

    df.to_csv(output, index=False)


def anomaly_clf(data, label, outliers_fraction=0.3):
    """
    Apply IsolationForest algorithm for anomaly detection, namely for euclidean distances
    that was calculated incorrectly because of incorrect predicted joints by the regressor model
    :param data: DataFrame object (~300 rows, 9 columns) of calculated 9 euclidean distances
    :param label: class name(integer) for each sign
    :param outliers_fraction: The amount of contamination of the data set, i.e. the proportion of outliers
                              in the data set. Used when fitting to define the threshold on the scores of the samples.
    :return: DataFrame, processed data using the IsolationForest.
    """
    clf = IsolationForest(contamination=outliers_fraction,
                          random_state=9)
    preds = clf.fit_predict(data)

    copied_data = data.copy(deep=True)
    copied_data['label'] = preds
    copied_data = copied_data[copied_data.label != -1]

    copied_data['label'] = label

    return copied_data


def predict_sign(joints, gesture_clf, classes):
    """
    Function for predicting a shown sign by passing as input array of 9 euclidean distances and get letter from
    classes dictionary for visualising.
    :param joints: (21, 2) array of predicted coordinates of joints
    :param gesture_clf: loaded Bayesian classifier model
    :param classes: dictionary of mapped labels {int: str}
    :return: string, predicted letter that represents a sign gesture
    """
    distances = calc_distances(joints)
    distances = np.expand_dims(distances, axis=0)
    pred = gesture_clf.predict(distances)[0]
    sign = classes[pred]

    return sign


def draw_points(points, frame):
    """
    Draw each point as a circle and connections between them as line
    :param points: (21, 2) array of predicted coordinates of joints
    :param frame: cv2 window
    :return:
    """
    for point in points:
        x, y = point
        cv2.circle(frame, (int(x), int(y)), THICKNESS*2, POINT_COLOR, THICKNESS)
    for connection in connections:
        x0, y0 = points[connection[0]]
        x1, y1 = points[connection[1]]
        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)


def draw_bboxes(bboxes, frame):
    """
    Draw bounding bbox of detected hand
    :param bboxes: 4 coordinates of bounding box
    :param frame: cv2 window
    :return:
    """
    if bboxes is not None:
        for i in range(0, len(bboxes)):
            i_inc_wrapped = (i + 1) % len(bboxes)
            start_x, start_y = bboxes[i]
            end_x, end_y = bboxes[i_inc_wrapped]
            cv2.line(frame, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0, 0, 255), THICKNESS)


def draw_sign(pred_sign, frame, position):
    """
    Draw predicted letter on cv2 frame
    :param pred_sign: predicted letter as string
    :param frame: cv2 window
    :param position: tuple of (x, y) where to put text
    :return:
    """
    word = "".join(pred_sign)
    cv2.putText(frame, word, position,
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, lineType=cv2.LINE_AA)