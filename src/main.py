""" main program """
import cv2
import time

class ImageData:

    def _image_cap(self):
        print("imagecap")
        self.cap = cv2.VideoCapture(0)
        time.sleep(1)
        _, frame = self.cap.read()
        
        return frame

class Display:

    def _image_show(self, image):
        print(image)
        self.image = image
        cv2.imshow("image", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    imagecap = ImageData()
    image = imagecap._image_cap()
    display = Display()
    display._image_show(image)

if __name__ == "__main__":
    main()

