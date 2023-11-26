import cv2
import numpy as np

# Open a connection to the webcam (usually 0 or 1, depending on your setup)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a window to display the webcam feed
cv2.namedWindow("Webcam Feed")

# Counter for naming the saved images
image_counter = 0

CHECKERBOARD = (5, 3) 
# stop the iteration when specified accuracy, epsilon, is reached or specified number of iterations are completed. 
criteria = (cv2.TERM_CRITERIA_EPS +
			cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
# Vector for 3D points 
threedpoints = [] 
# Vector for 2D points 
twodpoints = [] 
# 3D points real world coordinates 
objectp3d = np.zeros((1, CHECKERBOARD[0] 
					* CHECKERBOARD[1], 
					3), np.float32) 
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 
							0:CHECKERBOARD[1]].T.reshape(-1, 2) 
prev_img_shape = None
frames = []


while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow("Webcam Feed", frame)

    # Wait for the user to press a key (0xFF is a mask for 64-bit machines)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # add points if there is a checkers included
        grayColor = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        
	    # Find the chess board corners If desired number of corners are found in the image then ret = true 
        ret, corners = cv2.findChessboardCornersSB( 
	        grayColor, CHECKERBOARD, 
	        cv2.CALIB_CB_ADAPTIVE_THRESH 
	        + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_FILTER_QUADS
	        + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

        #cv2.imwrite('image_name.jpg', grayColor)
        print('found checkers corners?', ret)
        
        if ret: 
            threedpoints.append(objectp3d) 
            corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)
            twodpoints.append(corners2)  

            # Draw and display the corners 
            image = cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret) 

            image_counter += 1
            image_name = f"images/captured_image_{image_counter}.png"
            cv2.imwrite(image_name, frame)
            frames.append(frame)
            print(f"Image {image_name} saved.")

        if image_counter == 5:
            h, w = image.shape[:2] 
            # Perform camera calibration by 
            # passing the value of above found out 3D points (threedpoints) 
            # and its corresponding pixel coordinates of the 
            # detected corners (twodpoints) 
            ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera( 
            	threedpoints, twodpoints, grayColor.shape[::-1], None, None) 

            print(ret, matrix, distortion, r_vecs, t_vecs)
            break


    # If the 'q' key is pressed, exit the loop
    elif key == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
