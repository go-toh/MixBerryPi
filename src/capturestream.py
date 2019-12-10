import picamera
import numpy as np
import cv2
import io
import datetime
import time

#画像取得
def image_cap():
    camera.resolution = (3280, 1845)#16:9
    camera.capture(get_time()+".jpg")
    camera.resolution = (480, 270)
    time.sleep(0.1)

#時刻取得
def get_time():
    dt_time = datetime.datetime.now()
    nowtime = dt_time.strftime("%Y%m%d%H%M%S")
    return nowtime

#ストリーム取得
with picamera.PiCamera() as camera:
    camera.resolution = (480, 270)
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
        
        key = cv2.waitKey(100)
        print(key)
        #Enter Key
        if key == 13:
            image_cap()
        #Esc key
        if key == 27:
            break

cv2.destroyAllWindows()
