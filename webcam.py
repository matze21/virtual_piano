import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)

mpHands = mp.solutions.hands   #container to get hand key points from hand model

static_image_mode = False      # if true every image will be processed as a new frame, if False hand positions will be tracked over time
max_num_hands = 2
model_complexity = 1
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
hands = mpHands.Hands()#static_image_mode, model_complexity,max_num_hands, min_detection_confidence, min_tracking_confidence)

mpDraw = mp.solutions.drawing_utils # function to draw the points

tic = time.time()
tipPointsArray = [4,8,12,16,20]
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # transform image to color
    results = hands.process(imgRGB)               # run model
    h,w,c = img.shape # height width, channel

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for id,landmark in enumerate(hand.landmark):
                # id = handpoint
                # landmark = relative to img width/height coordinates
                if id in tipPointsArray:
                    #pixel values
                    cx,cy = int(landmark.x*w), int(landmark.y*h)

                    cv2.circle(img, (cx,cy), 25, (255,0,255), cv2.FILLED) #img, coordinates, radius, color, mode
            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)   #draw on original image, not rgb image

    #frame rate
    toc = time.time()
    fps = 1/(toc-tic)
    tic = time.time()
    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,2, (255,255,255),3) #scale, color, thickness


    cv2.imshow('Image',img)
    cv2.waitKey(1)