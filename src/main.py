""" main program """
import picamera
import io
import os
import datetime
import time
import cv2

#画像データ管理

#画像取得
def image_cap(time):
    with picamera.Picamera as camera:
        camera.resolution = (1024, 768)
        camera.start_preview()
        time.sleep(3)
        camera.capture(get_time()+".jpg")

#時刻取得
def get_time():
    dt_time = datetime.datetime.now()
    nowtime = dt_time.strftime("%Y%m%d%H%M%S")
    return nowtime

#main関数
def main():
    image_cap()
    
if __name__ == "__main__":
    main()
