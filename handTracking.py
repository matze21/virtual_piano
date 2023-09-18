import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
import keyboard

class handDetector():
    def __init__(self, mode=False,maxHands=2,detectionConf=0.5,trackConf=0.5):
        self.mode         =mode
        self.maxHands     =maxHands
        self.detectionConf=detectionConf
        self.trackConf    =trackConf
        
        self.tipPointsArray = [4,8,12,16,20] #ids of finger tips

        self.mpHands = mp.solutions.hands   #container to get hand key points from hand model
        self.hands   = self.mpHands.Hands(self.mode,int(self.maxHands),1, self.detectionConf, self.trackConf)
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
    
    def findTipPositions(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # transform image to color
        results = self.hands.process(imgRGB)          # run model
        h,w,c = img.shape # height width, channel
        tipArray = np.ones((10,3)) * np.nan

        tipCounter = 0
        if results.multi_hand_landmarks:
            for handId,hand in enumerate(results.multi_hand_landmarks):
                for id,landmark in enumerate(hand.landmark):
                    # id = handpoint
                    # landmark = relative to img width/height coordinates
                    if id in self.tipPointsArray:
                        #pixel values
                        idx = self.tipPointsArray.index(id)
                        idx = idx + 5 if handId > 0 else idx
                        cx,cy = int(landmark.x*w), int(landmark.y*h)
                        tipArray[idx, 0] = id
                        tipArray[idx, 1] = cx
                        tipArray[idx, 2] = cy
                        if draw:
                            cv2.circle(img, (cx,cy), 25, (255,0,255), cv2.FILLED) #img, coordinates, radius, color, mode
        
        return img, tipArray

    def renderPoint(self,img, cx,cy):
        cv2.circle(img, (cx,cy), 15, (255,100,255), cv2.FILLED) #img, coordinates, radius, color, mode
        return img


def main():
    cap = cv2.VideoCapture(1)

    hd = handDetector()
    tic = time.time()
    array = []
    columns=['4l','x4l','y4l','8l','x8l','y8l','12l','x12l','y12l','16l','x16l','y16l','20l','x20l','y20l',
             '4r','x4r','y4r','8r','x8r','y8r','12r','x12r','y12r','16r','x16r','y16r','20r','x20r','y20r']
    data = pd.DataFrame(array, columns=columns)
    
    y_feat = ['y4l','y8l','y12l','y16l','y20l','y4r','y8r','y12r','y16r','y20r']

    #try:
    if 1:
        while True:

            success, img = cap.read()
            img = cv2.flip(img, 1)
            img = hd.drawImg(img)
            img, tipArray = hd.findTipPositions(img, False)

            new_row_df = pd.DataFrame([tipArray.flatten()], columns=columns)
            data = pd.concat([data[columns], new_row_df], ignore_index=True, axis = 0)
            data = data.reset_index()

            if data.shape[0] > 10:
                data = data.drop(0, axis=0)

            for feat in y_feat:
                steigung = data[feat].rolling(3).mean().diff().iloc[-1]/data[feat].iloc[-1]
                if steigung > 0.02:

                    y = int(data[feat].iloc[-1])
                    x = int(data[feat.replace('y','x')].iloc[-1])

                    img = hd.renderPoint(img, x,y)

            #frame rate
            toc = time.time()
            fps = 1/(toc-tic)
            tic = time.time()
            cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,2, (255,255,255),3) #scale, color, thickness


            cv2.imshow('Image',img)
            cv2.waitKey(1)
    #except:
    #    print('stopped')
    keyboard.unhook_all()
    #data = pd.DataFrame(array, columns=['4l','x4l','y4l','8l','x8l','y8l','12l','x12l','y12l','16l','x16l','y16l','20l','x20l','y20l',
    #                                    '4r','x4r','y4r','8r','x8r','y8r','12r','x12r','y12r','16r','x16r','y16r','20r','x20r','y20r'])
    #data.to_csv('finger movement.csv')

if __name__ == '__main__':
    main()