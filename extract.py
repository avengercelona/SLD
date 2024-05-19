import mediapipe as mp
import os
import cv2
import numpy as np

def extrLandmarks(image_path):
    hands = mp.solutions.hands
    hands_mesh = hands.Hands(static_image_mode=True, min_detection_confidence=0.7) #hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_mesh.process(image_rgb)
    lm=[]
    for res in results.multi_hand_landmarks:
        for data_point in res.landmark:
            lm.extend([data_point.x, data_point.y, data_point.z])
    if len(lm) == 63:
        lm.extend([0]*63)
    lm=np.array(lm)
    return lm


'''hands = mp.solutions.hands
hands_mesh = hands.Hands(static_image_mode=True, min_detection_confidence=0.7) #hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
#draw = mp.solutions.drawing_utils
#mp_drawing = mp.solutions.drawing_utils # Drawing utilities

image_path = 'data/1/12.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = hands_mesh.process(image_rgb)
lm=[]
for res in results.multi_hand_landmarks:
    for data_point in res.landmark:
        lm.extend([data_point.x, data_point.y, data_point.z])
print(len(lm))
if len(lm) == 63:
    lm.extend([0]*63)
lm=np.array(lm)'''