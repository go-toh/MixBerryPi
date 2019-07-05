""" main program """
import cv2
import numpy
import time

def video_cap():
    capture = cv2.VideoCapture(0)
    time.sleep(1)
    ret, image = capture.read()
    return ret,image

ret, image = video_cap()

if ret == True:
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
else:
    print("ERROR!!")
