from tflite_runtime.interpreter import Interpreter
import picamera
import numpy as np
import cv2
import io
import time
import datetime

#画像取得
def image_cap(width, height):
    camera.resolution = (width, height)
    filepath = "image/" + get_time() + ".jpg"
    camera.capture(filepath)
    camera.resolution = (480, 270)
    return filepath

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

#物体検出の推論
def detect_objects(interpreter, image):
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))
    
    #推論結果をresult(dict)に成形しresults(list)に格納
    results = []
    for i in range(count):
        if scores[i] >= 0.5 and classes[i] == 0:#推論で60%以上の確率かつclassesがpersonの場合
            result = {
            'bounding_box': boxes[i],
            'class_id': classes[i],
            'score': scores[i],
            'label': LABELS[int(classes[i])]
            }
            results.append(result)
    return results

#resultsを元に画像の縦横に合わせた座標を計算.座標とscoreとlabelを抽出してlistで返す
def person_position(results, height, width):
    set_box = []
    for obj in results:
        score = str(round(obj['score'], 2))
        label = obj['label']
        ymin, xmin, ymax, xmax = obj['bounding_box']

        #負数になるのを防ぐために三項演算子を使っている...
        after_xmin = int(xmin * width) if int(xmin * width) >= 0 else 0
        after_ymax = int(ymax * height) if int(ymax * height) >= 0 else 0
        after_xmax = int(xmax * width) if int(xmax * width) >= 0 else 0
        after_ymin = int(ymin * height) if int(ymin * height) >= 0 else 0

        set_box.append([after_xmin, after_ymax, after_xmax, after_ymin, score, label])
    return set_box

def movement_detection():

    pass

if __name__ == '__main__':
    LABELS = ['person']
    person_flag = False

    interpreter = Interpreter("model/mobilenet_ssd_v2_coco_quant_postprocess.tflite")
    interpreter.set_num_threads(4)
    interpreter.allocate_tensors()

    #picameraの設定とlistの宣言
    with picamera.PiCamera() as camera:
        camera.resolution = (480, 270)
        camera.framerate = 15
        stream = io.BytesIO()
        path_array = []
        result_array = []

        #ストリームを取得してリアルタイム物体検出
        #480*270のストリームを300*300にresizeして推論し、480*270に復元してから表示
        while True:
            camera.capture(stream, format='jpeg', use_video_port=True, resize=(300,300))
            frame = np.frombuffer(stream.getvalue(), dtype=np.uint8)
            detectimage = cv2.imdecode(frame,1)
            results = detect_objects(interpreter, detectimage)
            resizeimage = cv2.resize(detectimage, (480, 270))

            #personが検出されたとき
            if results:
                stream_height, stream_width = resizeimage.shape[:2]
                box_position = person_position(results, stream_height, stream_width)

                if not person_flag:
                    for box in box_position:
                        left_position , right_position = box[0], box[2]

                    person_flag = True
                    print('person status')
                else:
                    cv2.line(resizeimage, (left_position,0), (left_position,stream_height), (0, 0, 255), 2)
                    cv2.line(resizeimage, (right_position,0), (right_position,stream_height), (0, 0, 255), 2)
                    for box in box_position:
                        left_position_after, right_position_after = box[0], box[2]
                        center_position = (left_position_after + right_position_after) / 2
                    
                    if not left_position < center_position < right_position:
                        result_array.append(results)
                        #path = image_cap(480, 270)#16:9
                        #path = image_cap(960, 540)#16:9
                        path = image_cap(3280, 1845)#16:9
                        path_array.append(path)
                        person_flag = False
                        print("capture")
                        
                #boxの描画
                for box in box_position:
                    cv2.line(resizeimage, (box[0],0), (box[0],stream_height), (0, 255, 0), 2) #left line
                    center_line = int((box[0] + box[2]) / 2)
                    cv2.line(resizeimage, (center_line,0), (center_line,stream_height), (255, 0, 0), 2) #center line
                    cv2.line(resizeimage, (box[2],0), (box[2],stream_height), (0, 255, 0), 2) #right line
                    cv2.putText(resizeimage, box[5]+' '+box[4], (box[0]+10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    print(f'{box[5]} ({box[0]},{box[1]}), ({box[2]},{box[3]}) score=>{box[4]}')

            #personが検出されなかったとき
            else:
                person_flag = False
                print('Nothing detected')

            cv2.imshow('image',resizeimage)
            stream.truncate()
            stream.seek(0)

            #キーコードを取得
            key = cv2.waitKey(1)

            #Press 'Enter' Key
            if key == 13:
                result_array.append(results)
                path = image_cap(3280, 1845)#16:9
                path_array.append(path)
                print("capture")

            #Press 'Esc' key
            if key == 27:
                print("end")
                break

    cv2.destroyAllWindows()

    #撮影した画像に物体検出を適用
    for img_num, (cap_path, cap_result) in enumerate(zip(path_array, result_array)):
        fullimage = cv2.imread(cap_path)
        cap_height, cap_width = fullimage.shape[:2]
        cap_box_position = person_position(cap_result, cap_height, cap_width)

        #boxの描画
        for box_num, box in enumerate(cap_box_position):

            #画像の切り出し.img_numが画像のインデックス.box_numが検出領域ごとのインデックス
            left_img = fullimage[0:cap_height, 0:box[0]]
            object_img = fullimage[0:cap_height, box[0]:box[2]]
            right_img = fullimage[0:cap_height, box[2]:cap_width]
            cv2.imwrite('image/' + str(img_num) + '_' + str(box_num) + '_'+ box[5] + 'left.jpg',left_img)
            cv2.imwrite('image/' + str(img_num) + '_' + str(box_num) + '_'+ box[5] + 'object.jpg',object_img)
            cv2.imwrite('image/' + str(img_num) + '_' + str(box_num) + '_'+ box[5] + 'right.jpg',right_img)

            #境界線とラベルの描画
            cv2.line(fullimage, (box[0],0), (box[0],cap_height), (0, 255, 0), 2) #left line
            full_center_line = int((box[0] + box[2]) / 2)
            cv2.line(fullimage, (full_center_line,0), (full_center_line,cap_height), (255, 0, 0), 2) #center line
            cv2.line(fullimage, (box[2],0), (box[2],cap_height), (0, 255, 0), 2) #right line
            cv2.putText(fullimage, box[5]+' '+box[4], (box[0]+10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f'{box[5]} ({box[0]},{box[1]}), ({box[2]},{box[3]}) score=>{box[4]}')

        #画像名の置換でresultを追加
        filename = cap_path.replace('image/', 'image/result')
        cv2.imwrite(filename, fullimage)
