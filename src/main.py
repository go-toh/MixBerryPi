""" main program """
import time
import cv2

#画像データ管理
class ImageData:

    #画像取得
    def _image_cap(self):
        self.cap = cv2.VideoCapture(0)
        time.sleep(1)
        _, frame = self.cap.read()
        return frame

#表示管理
class Display:

    #画像表示
    def _image_show(self, image):
        self.image = image
        cv2.imshow("image", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#main関数
def main():
    imagecap = ImageData()
    image = imagecap._image_cap()
    display = Display()
    display._image_show(image)

if __name__ == "__main__":
    main()
