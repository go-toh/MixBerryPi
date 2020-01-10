from tflite_runtime.interpreter import Interpreter
import picamera
import numpy as np
import cv2
import io
import time
import datetime
import threading

def wait_input():
    global key_flag
    input()
    key_flag = False

def set_interpreter(interpreter):
    interpreter.set_num_threads(4)
    interpreter.allocate_tensors()

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image
    
def get_output_tensor(interpreter, index):
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor

#物体検出の推論
def detect_objects(interpreter, image):
    detect_flag = False
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))
    
    scores_array = []
    
    for i in range(count):
        if scores[i] >= 0.5 and classes[i] == 0:
            scores_array.append(scores[i])
    
    if scores_array:
        max_score = scores_array.index(max(scores_array))
        target_box = boxes[max_score]
        detect_flag = True
    else:
        target_box = []
        detect_flag = False
        
    return detect_flag, target_box

def person_position(result, width):
    _, xmin, _, xmax  = result
    after_xmin = int(xmin * width) if int(xmin * width) >= 0 else 0
    after_xmax = int(xmax * width) if int(xmax * width) >= 0 else 0

    return after_xmin, after_xmax
    
def get_center_line(left, right):
    return (left + right) // 2

def image_cap(width, height, count):
    camera.resolution = (width, height)
    filepath = "image/" + str(count) + ".jpg"
    camera.capture(filepath)
    camera.resolution = (480, 270)
    return filepath

if __name__ == '__main__':
    interpreter = Interpreter("model/mobilenet_ssd_v2_coco_quant_postprocess.tflite")
    set_interpreter(interpreter)

    with picamera.PiCamera() as camera:
        image_width, image_height = 480,270
        camera.resolution = (image_width, image_height)
        camera.framerate = 15
        camera.shutter_speed = 30000
        camera.iso = 800

        stream = io.BytesIO()
        key_flag = True
        person_detect_flag = False
        push_count = 0

        th = threading.Thread(target=wait_input)
        th.start()

        while key_flag:
            camera.capture(stream, format='jpeg', use_video_port=True)
            frame = np.frombuffer(stream.getvalue(), dtype=np.uint8)
            getimage = cv2.imdecode(frame, 1)
            inputimage = cv2.resize(getimage, (300, 300))
            

            result_flag, result_box = detect_objects(interpreter, inputimage)

            if result_flag:
                left_line, right_line = person_position(result_box, image_width)
                center_line = get_center_line(left_line, right_line)
                #cv2.line(getimage, (left_line,0), (left_line,image_height), (0, 255, 0), 2)
                #cv2.line(getimage, (right_line,0), (right_line,image_height), (0, 255, 0), 2)
                #cv2.line(getimage, (center_line,0), (center_line,image_height), (255, 0, 0), 2)
                print(left_line, right_line, center_line)

                if not person_detect_flag:
                    save_left_line, save_right_line = left_line, right_line
                    person_detect_flag = True
                else:
                    pass
                    #cv2.line(getimage, (save_left_line,0), (save_left_line,image_height), (0, 0, 255), 2)
                    #cv2.line(getimage, (save_right_line,0), (save_right_line,image_height), (0, 0, 255), 2)
                    
                if not save_left_line < center_line < save_right_line:
                    push_count += 1
                    print(push_count)
                    file_path = image_cap(960, 540, push_count)
                    person_detect_flag = False

            else:
                print('Not detection')

            #cv2.imshow('image', getimage)
            #cv2.waitKey(1)
            stream.truncate()
            stream.seek(0)

    th.join()
    cv2.destroyAllWindows()    
