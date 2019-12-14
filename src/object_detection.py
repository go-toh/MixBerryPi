from tflite_runtime.interpreter import Interpreter
import picamera
import numpy as np
import cv2
import io
import time
import datetime

#画像取得
def image_cap():
    camera.resolution = (3280, 1845)#16:9
    camera.capture("image/"+get_time()+".jpg")
    camera.resolution = (480, 270)

#時刻取得
def get_time():
    dt_time = datetime.datetime.now()
    nowtime = dt_time.strftime("%Y%m%d%H%M%S")
    return nowtime

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image
    
def get_output_tensor(interpreter, index):
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor

def detect_objects(interpreter, image):
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))
    
    results = []
    for i in range(count):
        #推論で50%以上の確率の場合
        if scores[i] >= 0.5:
            result = {
            'bounding_box': boxes[i],
            'class_id': classes[i],
            'score': scores[i],
            'label': LABELS[int(classes[i]+1)]
            }
            results.append(result)
    return results

def draw_box(results, height, width):
    for obj in results:
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * width)
        xmax = int(xmax * width)
        ymin = int(ymin * height)
        ymax = int(ymax * height)
        score = str(round(obj['score'], 2))
        label = obj['label']
        cv2.rectangle(showimage, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)
        cv2.putText(showimage, label+' '+score, (xmin,ymin-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f'{label} ({xmin},{ymax}), ({xmax},{ymin}) score=>{score}')

LABELS = [
'???','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
'traffic light','fire hydrant','???','stop sign','parking meter','bench','bird','cat','dog','horse',
'sheep','cow','elephant','bear','zebra','giraffe','???','backpack','umbrella','???',
'???','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
'baseball glove','skateboard','surfboard','tennis racket','bottle','???','wine glass','cup','fork','knife',
'spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
'donut','cake','chair','couch','potted plant','bed','???','dining table','???','???',
'toilet','???','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
'toaster','sink','refrigerator','???','book','clock','vase','scissors','teddy bear','hair drier',
'toothbrush']

if __name__ == '__main__':
    interpreter = Interpreter("model/mobilenet_ssd_v2_coco_quant_postprocess.tflite")
    interpreter.allocate_tensors()
    #ストリーム取得
    with picamera.PiCamera() as camera:
        camera.resolution = (480, 270)
        camera.framerate = 30
        height, width = 270, 480
        stream = io.BytesIO()
        interpreter = Interpreter("model/mobilenet_ssd_v2_coco_quant_postprocess.tflite")
        interpreter.allocate_tensors()

        while True:
            camera.capture(stream, format='jpeg', use_video_port=True, resize=(300,300))
            frame = np.frombuffer(stream.getvalue(), dtype=np.uint8)
            image = cv2.imdecode(frame,1)
            results = detect_objects(interpreter, image)
            showimage = cv2.resize(image, (480, 270))
            draw_box(results, height, width)
            cv2.imshow('image',showimage)
            stream.truncate()
            stream.seek(0)

            key = cv2.waitKey(1)
            #Enter Key
            if key == 13:
                image_cap()
                print("capture")
            #Esc key
            if key == 27:
                print("end")
                break
    cv2.destroyAllWindows()

