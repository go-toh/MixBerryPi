import picamera
import numpy as np
import cv2
import io

with picamera.PiCamera() as camera:
    camera.resolution = (320,240)
    camera.framerate = 30
    stream = io.BytesIO()
    key=0

    while True:
        camera.capture(stream, format='jpeg', use_video_port=True)
        frame = np.frombuffer(stream.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(frame,1)
        cv2.imshow('image',image)
        stream.truncate()
        stream.seek(0)
        
        #Esc key
        key = cv2.waitKey(100)
        if key == 27:
            break

cv2.destroyAllWindows()
