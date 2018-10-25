import re
import cv2
import json
import matplotlib.pyplot as plt

if __name__ == '__main__':
    image_num = 0
    with open('./face_result.txt') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if len(line) > 0:
                image_file = './live_image/' + str(image_num) + '.jpg'
                print('image_file: ', image_file)
                image = cv2.imread(image_file)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = json.loads(line)
                output = result['output']
                rect_angle = output[0]['face_rectangle']
                if len(rect_angle) > 2:
                    rect = rect_angle.replace('[', '').replace(']','')
                    rect = rect.split(',')
                    h, w, _ = image.shape
                    rect_num = len(rect) // 4
                    for index in range(rect_num):
                        cv2.rectangle(image, (int(float(rect[0 + index * 4]) * w), int(float(rect[1 + index * 4]) * h)), (int(float(rect[2 + index *4]) * w), int(float(rect[3 + index * 4]) * h)), (0, 155, 255), 2)
                cv2.imwrite('./face_feature/'+ str(image_num) + '.jpg', image)
                image_num += 1