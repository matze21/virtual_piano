
import cv2
import torch
import matplotlib.pyplot as plt

# result: resolution is not good enough for separating fingers


midas = torch.hub.load("intel-isl/MiDaS","MiDaS_small")
midas.to('cpu')
midas.eval() #removes batch normalization,.. for training

transforms = torch.hub.load("intel-isl/MiDaS","transforms")
transform = transforms.small_transform

cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    imgBatch = transform(img).to('cpu') # feature space for midas model
    with torch.no_grad():
        prediciton = midas(imgBatch)
        prediciton = torch.nn.functional.interpolate(
            prediciton.unsqueeze(1),
            size = img.shape[:2],
            mode = 'bicubic',
            align_corners = False).squeeze()
        
        output = prediciton.cpu().numpy()

        plt.imshow(output)
        plt.pause(0.0000001)


    cv2.imshow('Image',img)
    if cv2.waitKey(1) & 0xFF == ord('q'): #terminate loop
        cap.release()
        cv2.destroyAllWindows()

plt.show()