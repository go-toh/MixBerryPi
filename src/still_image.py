from tflite_runtime.interpreter import Interpreter
import picamera
import numpy as np
import cv2
import time

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
        if scores[i] >= 0.5:
            result = {
            'bounding_box': boxes[i],
            'class_id': classes[i],
            'score': scores[i],
            'label': LABELS[int(classes[i])]
            }
            results.append(result)
    return results

def draw_box(results, height, width):
    for index, obj in enumerate(results):
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * width)
        xmax = int(xmax * width)
        ymin = int(ymin * height)
        ymax = int(ymax * height)
        score = str(round(obj['score'], 2))
        label = obj['label']
        cv2.rectangle(image, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)
        cv2.putText(image, label+' '+score, (xmin,ymin-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f'{index} {label} ({xmin},{ymax}), ({xmax},{ymin}) score=>{score}')

LABELS = [
'???''person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
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
    image = cv2.imread("image/dog.jpg")
    height, width = image.shape[:2]
    detectimage = cv2.resize(image, (300, 300))
    interpreter = Interpreter("model/mobilenet_ssd_v2_coco_quant_postprocess.tflite")
    interpreter.allocate_tensors()
    results = detect_objects(interpreter, detectimage)
    
    if results:
        draw_box(results, height, width)
        cv2.imwrite('image/result.jpg',image)
    