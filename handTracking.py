import cv2
import mediapipe as mp
import time
import numpy as np


class handDetector():
    def __init__(self, mode=False,maxHands=2,detectionConf=0.5,trackConf=0.5):
        self.mode         =mode
        self.maxHands     =maxHands
        self.detectionConf=detectionConf
        self.trackConf    =trackConf
        
        self.tipPointsArray = [4,8,12,16,20] #ids of finger tips

        self.mpHands = mp.solutions.hands   #container to get hand key points from hand model
        self.hands   = self.mpHands.Hands(self.mode,self.maxHands, self.detectionConf, self.trackConf)
        self.mpDraw  = mp.solutions.drawing_utils # function to draw the points


    def drawImg(self, img, drawLandmarks = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # transform image to color
        results = self.hands.process(imgRGB)               # run model
        h,w,c = img.shape # height width, channel

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                if drawLandmarks:
                    self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)   #draw on original image, not rgb image

        return img
    
    def findTipPositions(self, img, handNr = 0, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # transform image to color
        results = self.hands.process(imgRGB)          # run model
        h,w,c = img.shape # height width, channel
        tipArray = np.ones((10,3)) * np.nan

        tipCounter = 0
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                for id,landmark in enumerate(hand.landmark):
                    # id = handpoint
                    # landmark = relative to img width/height coordinates
                    if id in self.tipPointsArray:
                        #pixel values
                        cx,cy = int(landmark.x*w), int(landmark.y*h)
                        tipArray[tipCounter, 0] = id
                        tipArray[tipCounter, 1] = cx
                        tipArray[tipCounter, 2] = cy
                        if draw:
                            cv2.circle(img, (cx,cy), 25, (255,0,255), cv2.FILLED) #img, coordinates, radius, color, mode
        
        return img, tipArray


        







def main():
    cap = cv2.VideoCapture(1)
    handDetector = handDetector()

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = handDetector.drawImg()

    
        #frame rate
        toc = time.time()
        fps = 1/(toc-tic)
        tic = time.time()
        cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,2, (255,255,255),3) #scale, color, thickness
        tic = time.time()

        cv2.imshow('Image',img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()