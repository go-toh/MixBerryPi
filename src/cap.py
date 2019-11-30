""" main program """
import picamera
import datetime
import time

#画像取得
def image_cap(nowtime):
    with picamera.PiCamera() as camera:
        camera.resolution = (1024, 768)
        camera.start_preview()
        time.sleep(3)
        camera.capture(nowtime+".jpg")

#時刻取得
def get_time():
    dt_time = datetime.datetime.now()
    nowtime = dt_time.strftime("%Y%m%d%H%M%S")
    return nowtime

#main関数
def main():
    gettime = get_time()
    image_cap(gettime)
    
if __name__ == "__main__":
    main()
