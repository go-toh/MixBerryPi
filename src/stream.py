import picamera
import cv2
import numpy as np
import time
import io

stream = io.BytesIO()
key = 0
with picamera.PiCamera() as camera:
    for i in camera.capture_continuous(stream,format='jpeg',use_video_port=True,resize=(320,240)):
        stream.truncate()
        stream.seek(0)
        data = np.frombuffer(stream.getvalue(),dtype=np.uint8)
        image = cv2.imdecode(data,1)
        cv2.imshow('image',image)
        key = cv2.waitKey(100)
        
        #Escキーで終了
        if key == 27:
            break
    
cv2.destroyAllWindows()
    



    

