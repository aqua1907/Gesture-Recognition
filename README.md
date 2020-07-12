# Hand tracker and ASL recognition

This work is based on Google's work [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands)

Thanks to:
 - [wolterlw](https://github.com/wolterlw/hand_tracking) for script that runs model itself using `TensorFlow Lite Interpreter`;
 - [metalwhale](https://github.com/metalwhale/hand_tracking) for `palm_detection_without_custom_op.tflite` and visualization results;
 - [FabianHertwig](https://github.com/FabianHertwig/hand_tracking) for using Non-Max-Suppression algorithm, implemented by Adrian Rosebrock;
 
## How does it work?
Palm is recognized by BlazePalm Detector and then cropping 256x256 region for prediction 2-D or 3-D coordinates of hand landmark. Using this landmarks I calculate 9 Euclidean distances between:
(20, 0), (16, 0), (12, 0), (8, 0), (4, 0), (20, 16), (16, 12), (12, 8), (8, 4).
![](https://github.com/aqua1907/hand_landmark/blob/master/images/photo_2020-04-28_15-19-47.jpg)

Combination of 9 euclidean distances represent a letter of sign language alphabet. Bayesian classifier predicts a shown sign based on a combination of 9 euclidean distances.
![Result](https://github.com/aqua1907/hand_landmark/blob/master/images/hand_landmark_flex.mp4?raw=True "Result")

## Files
- `explore_data.ipynb` — Inspect raw key points data and apply Isolation forest. [Link](explore_data.ipynb)
- `train_model.ipynb` — Train Bayesian classifier on prepared data. [Link](run_ipynb.ipynb)
- `run.py` — run script on video feed. [Link](run.py)
- `run_on_image.py` — run script on image. [Link](run_on_image.py)
