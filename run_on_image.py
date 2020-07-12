import cv2
from src.hand_tracker_nms import HandTrackerNMS
import src.extra
import joblib


WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"

connections = src.extra.connections
int_to_char = src.extra.classes

detector = HandTrackerNMS(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)

gesture_clf = joblib.load(r'models\\gesture_clf.pkl')

cv2.namedWindow(WINDOW)

word = []
letter = ""
staticGesture = 0

image_orig = cv2.imread(r'input//my_Y.jpg')
image_orig = cv2.resize(image_orig, (400, 600))
image = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
points, bboxes, joints = detector(image)

if points is not None:
    src.extra.draw_points(points, image_orig)
    pred_sign = src.extra.predict_sign(joints, gesture_clf, int_to_char)
    src.extra.draw_sign(pred_sign, image_orig, (50, 560))

cv2.imshow(WINDOW, image_orig)

cv2.waitKey(0)
cv2.destroyAllWindows()
