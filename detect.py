import cv2
import matplotlib.pyplot as plt
import os
from mtcnn import mtcnn
import time

def detect_face(weights_file, image_file):
    """
    Introduction
    ------------
        使用mtcnn模型检测人脸
    Parameters
    ----------
        weights_file: 模型权重文件
        image_file: 检测图片文件
    Returns
    -------
        result: 检测结果
    """

    detector = mtcnn(weights_file)
    start = time.time()
    files = os.listdir(image_file)
    for file in files:
        print('image_file', image_file + file)
        image = cv2.imread(image_file + file)
        result = detector.detect_face(image)
        print('detect time: {}'.format(time.time() - start))
        print(result)
        if len(result) > 0:
            for bbox in result:
                bounding_box = bbox['box']
                cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),(0,155,255), 2)
        # plt.imshow(image)
        # plt.show()
        cv2.imwrite('./result_mtcnn/' + file , image)


if __name__ == '__main__':
    detect_face('./mtcnn_weights.npy', './live_image/')