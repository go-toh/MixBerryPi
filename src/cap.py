""" main program """
import picamera
import datetime
import time

#画像取得
def image_cap():
    with picamera.PiCamera() as camera:
        camera.resolution = (3280, 2464)#MAX解像度
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
