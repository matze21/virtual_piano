import cv2
import mediapipe as mp
import time
import numpy as np
import pandas as pd
from scipy.optimize import fsolve


class handDetector():
    def __init__(self, mode=False,maxHands=2,detectionConf=0.5,trackConf=0.5):
        self.mode         =mode
        self.maxHands     =maxHands
        self.detectionConf=detectionConf
        self.trackConf    =trackConf
        self.solutions    = []
        self.xGridVal = []
        self.yGridVal = []
        self.gridAvailable = False
        self.handDistFeatures = {}
        
        self.tipPointsArray = [4,8,12,16,20, 3,7,11,15,19] #ids of finger tips
        self.tips     = [4,8,12,16,20]
        self.knuckles = [3,7,11,15,19]

        self.mpHands = mp.solutions.hands   #container to get hand key points from hand model
        self.hands   = self.mpHands.Hands(self.mode,int(self.maxHands),1, self.detectionConf, self.trackConf)
        self.mpDraw  = mp.solutions.drawing_utils # function to draw the points
        self.calibrationData = {}


    def drawImg(self, img, drawLandmarks = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # transform image to color
        results = self.hands.process(imgRGB)               # run model
        h,w,c = img.shape # height width, channel

        # render contours - does not help
        if 0:
            #Now convert the grayscale image to binary image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)        
            #Now detect the contours
            contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)           
            # draw contours on the original image
            img = cv2.drawContours(img, contours, -1, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)


        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                if drawLandmarks:
                    self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)   #draw on original image, not rgb image

        return img
    
    def findTipPositions(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # transform image to color
        results = self.hands.process(imgRGB)          # run model
        h,w,c = img.shape # height width, channel
        tipArray = np.ones((int(2*len(self.tipPointsArray)),3)) * np.nan

        tipCounter = 0
        if results.multi_hand_landmarks:
            for handId,hand in enumerate(results.multi_hand_landmarks):
                for id,landmark in enumerate(hand.landmark):
                    # id = handpoint
                    # landmark = relative to img width/height coordinates
                    if id in self.tipPointsArray:
                        #pixel values
                        idx = self.tipPointsArray.index(id)
                        idx = idx + len(self.tipPointsArray) if handId > 0 else idx
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

    def calcZEgo(self, x, y):
        yc,fl,pitch,zCam1,zCam2,zCam3,zCam4,zCam5 = x 
        eq1 = (np.sin(pitch) - (y - yc) / fl * np.cos(pitch)) * zCam1
        return eq1
    
    def calcZEgoExt(self, y):
            yc,fl,pitch,zCam1,zCam2,zCam3,zCam4,zCam5 = self.solutions
            eq1 = (np.sin(pitch) - (y - yc) / fl * np.cos(pitch)) * zCam1
            return eq1

    def calcZCam(self,x,y):
        yc,fl,pitch,zCam1,zCam2,zCam3,zCam4,zCam5 = x 
        eq1 = self.zEgo / (np.sin(pitch) - (y - yc) / fl * np.cos(pitch))
        return eq1


    def functions(self,x):
        yc,fl,pitch,zCam1,zCam2,zCam3,zCam4,zCam5 = x 
        eq1 = (np.sin(pitch) - (self.y1 - yc) / fl * np.cos(pitch)) * zCam1 - (np.sin(pitch) - (self.y2 - yc) / fl * np.cos(pitch)) * zCam2
        eq2 = (np.sin(pitch) - (self.y1 - yc) / fl * np.cos(pitch)) * zCam1 - (np.sin(pitch) - (self.y3 - yc) / fl * np.cos(pitch)) * zCam3
        eq3 = (np.sin(pitch) - (self.y1 - yc) / fl * np.cos(pitch)) * zCam1 - (np.sin(pitch) - (self.y4 - yc) / fl * np.cos(pitch)) * zCam4
        eq4 = (np.sin(pitch) - (self.y1 - yc) / fl * np.cos(pitch)) * zCam1 - (np.sin(pitch) - (self.y5 - yc) / fl * np.cos(pitch)) * zCam5
        eq5 = (np.sin(pitch) - (self.y2 - yc) / fl * np.cos(pitch)) * zCam2 - (np.sin(pitch) - (self.y3 - yc) / fl * np.cos(pitch)) * zCam3
        eq6 = (np.sin(pitch) - (self.y2 - yc) / fl * np.cos(pitch)) * zCam2 - (np.sin(pitch) - (self.y4 - yc) / fl * np.cos(pitch)) * zCam4
        eq7 = (np.sin(pitch) - (self.y2 - yc) / fl * np.cos(pitch)) * zCam2 - (np.sin(pitch) - (self.y5 - yc) / fl * np.cos(pitch)) * zCam5
        eq8 = (np.sin(pitch) - (self.y3 - yc) / fl * np.cos(pitch)) * zCam3 - (np.sin(pitch) - (self.y4 - yc) / fl * np.cos(pitch)) * zCam4
        #eq9 = (np.sin(pitch) - (y3 - yc) / fl * np.cos(pitch)) * zCam3 - (np.sin(pitch) - (y5 - yc) / fl * np.cos(pitch)) * zCam5
        #eq10 = (np.sin(pitch) - (y4 - yc) / fl * np.cos(pitch)) * zCam4 - (np.sin(pitch) - (y5 - yc) / fl * np.cos(pitch)) * zCam5
        return [eq1, eq2, eq3, eq4,eq5,eq6,eq7,eq8]#,eq9,eq10]

    def findCalibration(self, data : pd.DataFrame):
        vals = data.mean(axis=0)
        self.y_vals = {}
        for i,index in enumerate(vals.index):
            if 'y' in index:
                self.y_vals[index] = vals[i]
        self.y5 = self.y_vals['4.0yr']
        self.y1 = self.y_vals['8.0yr']
        self.y2 = self.y_vals['12.0yr']
        self.y3 = self.y_vals['16.0yr']
        self.y4 = self.y_vals['20.0yr']
        print(self.y_vals)

        solutions= fsolve(self.functions, [300, 600,np.deg2rad(-30),30,30,30,30,33])
        print('solution: ', solutions)
        print('solution accuracy', self.functions(solutions))
        zEgo1 = self.calcZEgo(solutions, self.y1)
        zEgo2 = self.calcZEgo(solutions, self.y2)
        zEgo3 = self.calcZEgo(solutions, self.y3)
        zEgo4 = self.calcZEgo(solutions, self.y4)
        zEgo5 = self.calcZEgo(solutions, self.y5)
        print('zEgos',zEgo1, zEgo2,zEgo3,zEgo4,zEgo5)
        self.solutions = solutions
        self.zEgo = zEgo1

        # relative finger size
        self.handDistFeatures['dist4.0r']  = self.calcRelativeFingerSize('4.0yr','3.0yr',vals)
        self.handDistFeatures['dist8.0r']  = self.calcRelativeFingerSize('8.0yr','7.0yr',vals)
        self.handDistFeatures['dist12.0r'] = self.calcRelativeFingerSize('12.0yr','11.0yr',vals)
        self.handDistFeatures['dist16.0r'] = self.calcRelativeFingerSize('16.0yr','15.0yr',vals)
        self.handDistFeatures['dist20.0r'] = self.calcRelativeFingerSize('20.0yr','19.0yr',vals)
        self.handDistFeatures['dist4.0l']  = self.calcRelativeFingerSize('4.0yl','3.0yl',vals)
        self.handDistFeatures['dist8.0l']  = self.calcRelativeFingerSize('8.0yl','7.0yl',vals)
        self.handDistFeatures['dist12.0l'] = self.calcRelativeFingerSize('12.0yl','11.0yl',vals)
        self.handDistFeatures['dist16.0l'] = self.calcRelativeFingerSize('16.0yl','15.0yl',vals)
        self.handDistFeatures['dist20.0l'] = self.calcRelativeFingerSize('20.0yl','19.0yl',vals)
        print(self.handDistFeatures)
    
        # key 1 has to be the bottom one!!
        # the keys have to be the y keys!
    def calcRelativeFingerSize(self, key1, key2, vals):
        y = vals[key1]
        sizeInImage = np.sqrt((vals[key1]-vals[key2])**2 +(vals[key1.replace('x','y')]-vals[key2.replace('x','y')])**2)
        zCamFlatWorld = self.calcZCam(self.solutions, y)
        return zCamFlatWorld/self.solutions[1]*sizeInImage
    
    def calcZCamFromFingerSize(self,key,imgDist):
        return self.handDistFeatures[key]*self.solutions[1]/imgDist
    
    def calcImgXY(self, xEgo,zEgo,yEgo,xc):
        yc,fl,pitch,zCam1,zCam2,zCam3,zCam4,zCam5 = self.solutions
        yCam = (xEgo-zEgo/np.tan(pitch))/(np.cos(pitch)**2/np.sin(pitch)+np.sin(pitch))
        zCam = (zEgo + np.cos(pitch)*yCam)/np.sin(pitch)

        yImg = yCam/zCam * fl + yc
        xImg = yEgo/zCam * fl + xc
        return xImg,yImg

    def calcGrid(self, img):
        x_interval = np.linspace(0,100,20)
        y_interval = np.linspace(-50,50,10)
        xc = img.shape[1]/2

        x_vals = np.ones((20,10))*np.nan
        y_vals = np.ones((20,10))*np.nan

        for i,x in enumerate(x_interval):
            for j,y in enumerate(y_interval):
                x_vals[i,j], y_vals[i,j] = self.calcImgXY(x,self.zEgo,y,xc)
        self.xGridVal = x_vals
        self.yGridVal = y_vals
        self.gridAvailable = True
        
    def renderGrid(self,img):
        if self.gridAvailable:
            for i in range(20):
                for j in range(10):
                    if i+1 < 20:
                        cv2.line(img, (int(self.xGridVal[i,j]),int(self.yGridVal[i,j])), (int(self.xGridVal[i+1,j]),int(self.yGridVal[i+1,j])), color=(0, 255, 0), thickness=2)  # Draw green lines between points
                    if j+1 < 10:
                        cv2.line(img, (int(self.xGridVal[i,j]),int(self.yGridVal[i,j])), (int(self.xGridVal[i,j+1]),int(self.yGridVal[i,j+1])), color=(0, 255, 0), thickness=2)
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

    log = pd.DataFrame()
    if 1:
        startT = time.time()
        CalibDone = False
        calibDataList = []
        while cap.isOpened():
            
            success, img = cap.read()
            img = cv2.flip(img, 1)
            img = hd.drawImg(img)

            #frame rate
            toc = time.time()
            fps = 1/(toc-tic)
            tic = time.time()
            cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,2, (255,255,255),3) #scale, color, thickness

            # introduction
            deltaTime1 = 10
            if time.time() < (startT + deltaTime1):
                diff = deltaTime1 - int(time.time() - startT)
                cv2.putText(img,'put your fingers on the surface for 10 seconds after the timer is up: '+str(int(diff)),(30,500), cv2.FONT_HERSHEY_PLAIN,2, (255,0,255),3)

            # calibration data gathering
            deltaTime2 = deltaTime1 + 15
            idx = 0
            if time.time() < (startT + deltaTime2) and time.time() > (startT + deltaTime1):
                diff = int(deltaTime2 - (time.time() - startT))
                cv2.putText(img,'calibrating.. hold still!'+str(int(diff)),(30,100), cv2.FONT_HERSHEY_PLAIN,2, (255,0,255),3)
                img, tipArray = hd.findTipPositions(img, True)
                row = {}
                for i,id in enumerate(tipArray[:,0]):
                    if not np.isnan(id):
                        if (str(id)+'xr') in row:
                            row[str(id)+'xl'] = tipArray[i,1]
                            row[str(id)+'yl'] = tipArray[i,2]
                        else:
                            row[str(id)+'xr'] = tipArray[i,1]
                            row[str(id)+'yr'] = tipArray[i,2]

                calibDataList.append(row)
                idx = idx+1

            # calc calibration
            if time.time() > (startT + deltaTime2) and not CalibDone:
                dat = pd.DataFrame(calibDataList)
                dat.to_csv("calibration.csv")
                hd.findCalibration(dat)
                CalibDone = True
                hd.calcGrid(img)

            img = hd.renderGrid(img)

            # check if finger hits surface
            if CalibDone:
                img, tipArray = hd.findTipPositions(img, False)
                for i,id in enumerate(tipArray[:,0]):
                    if not np.isnan(id):
                        suffix = 'l' if i < 10 else 'r'
                        key = 'dist'+str(id)+suffix
                        if id in hd.tips and (not np.isnan(tipArray[i+5,0])) and (tipArray[i+5,0] in hd.knuckles): #find knuckle
                            dist = np.sqrt((tipArray[i,1]-tipArray[i+5,1])**2 +(tipArray[i,2]-tipArray[i+5,2])**2)

                            zCamFinger = hd.calcZCamFromFingerSize(key,dist)
                            zEgo = hd.calcZEgoExt(zCamFinger)

                            if abs(zEgo - hd.zEgo) < 5:
                                print(zEgo, hd.Ego, id, key)
                                cv2.circle(img, (int(tipArray[i,1]),int(tipArray[i,2])), 25, (0,0,255), cv2.FILLED)


            cv2.imshow('Image',img)
            cv2.waitKey(1)

            if cv2.waitKey(1) & 0xFF == ord('q'): #terminate loop
                cap.release()
                cv2.destroyAllWindows()
    #data = pd.DataFrame(array, columns=['4l','x4l','y4l','8l','x8l','y8l','12l','x12l','y12l','16l','x16l','y16l','20l','x20l','y20l',
    #                                    '4r','x4r','y4r','8r','x8r','y8r','12r','x12r','y12r','16r','x16r','y16r','20r','x20r','y20r'])
    
    log.to_csv('finger movement.csv')

if __name__ == '__main__':
    main()