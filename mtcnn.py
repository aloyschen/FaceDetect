import cv2
import tensorflow as tf
import numpy as np
from layers import conv, prelu, max_pool, fc
from tensorflow.python.tools.freeze_graph import freeze_graph


class PNet:
    def __init__(self, session):
        self.session = session
        self.__build_net()

    def __build_net(self):
        """
        Introduction
            构建mtcnn模型级联第一层
        """
        with tf.variable_scope('pnet'):
            self.input = tf.placeholder(name = 'input_data', shape = [None, None, None, 3], dtype = tf.float32)
            layer = conv('conv1', self.input, kernel_size = (3, 3), channels_output = 10, stride = (1, 1), padding = 'VALID', relu = False)
            layer = prelu('prelu1', layer)
            layer = max_pool('pool1', layer, kernel_size = [2, 2], stride = (2, 2))
            layer = conv('conv2', layer, kernel_size = (3, 3), channels_output = 16, stride = (1, 1), padding = 'VALID', relu = False)
            layer = prelu('prelu2', layer)
            layer = conv('conv3', layer, kernel_size = (3, 3), channels_output = 32, stride = (1, 1), padding = 'VALID', relu = False)
            layer = prelu('prelu3', layer)
            conv4_1 = conv('conv4-1', layer, kernel_size = (1, 1), channels_output = 2, stride = (1, 1), relu = False)
            self.prob = tf.nn.softmax(conv4_1, axis = 3, name = 'prob')
            self.loc = conv('conv4-2', layer, kernel_size = (1, 1), channels_output = 4, stride = (1, 1), relu = False)


    def feed(self, image):
        """
        Introduction
        ------------
            返回pnet模型结构计算结果
        Parameters
        ----------
            image: 输入图片
        Returns
        -------
            prob: 是否为人脸的概率值
            loc: 人脸位置坐标
        """
        return self.session.run([self.loc, self.prob], feed_dict = {self.input : image})


class RNet:
    def __init__(self, session):
        self.session = session
        self.__build_net()

    def __build_net(self):
        """
        Introduction
        ------------
            构建RNet模型网络结构
        """
        with tf.variable_scope('rnet'):
            self.input = tf.placeholder(name = 'input_data', shape = [None, 24, 24, 3], dtype = tf.float32)
            layer = conv('conv1', self.input, kernel_size = (3, 3), channels_output = 28, stride = (1, 1), padding = 'VALID', relu = False)
            layer = prelu('prelu1', layer)
            layer = max_pool('pool1', layer, kernel_size = (3, 3), stride = (2, 2))
            layer = conv('conv2', layer, kernel_size = (3, 3), channels_output = 48, stride = (1, 1), padding = 'VALID', relu = False)
            layer = prelu('prelu2', layer)
            layer = max_pool('pool2', layer, kernel_size = (3, 3), stride = (2, 2), padding = 'VALID')
            layer = conv('conv3', layer, kernel_size = (2, 2), channels_output = 64, stride = (1, 1), padding = 'VALID', relu = False)
            layer = prelu('prelu3', layer)
            layer = fc('fc1', layer, channels_output = 128, relu = False)
            layer = prelu('prelu4', layer)
            fc2 = fc('fc2-1', layer, channels_output = 2, relu = False)
            self.prob = tf.nn.softmax(fc2, axis = 1, name = 'prob')
            self.loc = fc('fc2-2', layer, channels_output = 4, relu = False)


    def feed(self, image):
        """
        Introduction
        ------------
            返回pnet模型结构计算结果
        Parameters
        ----------
            image: 输入图片
        Returns
        -------
            prob: 是否为人脸的概率值
            loc: 人脸位置坐标
        """
        return self.session.run([self.loc, self.prob], feed_dict = {self.input : image})


class ONet:
    def __init__(self, session):
        self.session = session
        self.__build_net()

    def __build_net(self):
        """
        Introduction
        ------------
            构建ONet模型结构
        """
        with tf.variable_scope('onet'):
            self.input = tf.placeholder(shape = [None, 48, 48, 3], dtype = tf.float32, name = 'input_data')
            layer = conv('conv1', self.input, kernel_size = (3, 3), channels_output = 32, stride = (1,1), padding = 'VALID', relu = False)
            layer = prelu('prelu1', layer)
            layer = max_pool('pool1', layer, kernel_size = (3, 3), stride = (2, 2))
            layer = conv('conv2', layer, kernel_size = (3, 3), channels_output = 64, stride = (1, 1), padding = 'VALID', relu = False)
            layer = prelu('prelu2', layer)
            layer = max_pool('pool2', layer, kernel_size = (3, 3), stride = (2, 2), padding = 'VALID')
            layer = conv('conv3', layer, kernel_size = (3, 3), channels_output = 64, stride = (1, 1), padding = 'VALID', relu = False)
            layer = prelu('prelu3', layer)
            layer = max_pool('pool3', layer, kernel_size = (2, 2), stride = (2, 2))
            layer = conv('conv4', layer, kernel_size = (2, 2), channels_output = 128, stride = (1, 1), padding = 'VALID', relu = False)
            layer = prelu('prelu4', layer)
            layer = fc('fc1', layer, channels_output = 256, relu = False)
            layer = prelu('prelu5', layer)
            fc2 = fc('fc2-1', layer, channels_output = 2, relu = False)
            self.prob = tf.nn.softmax(fc2, axis = 1, name ='prob')
            self.loc = fc('fc2-2', layer, channels_output = 4, relu = False)


    def feed(self, image):
        """
        Introduction
        ------------
            返回onet模型结构计算结果
        Parameters
        ----------
            image: 输入图片
        Returns
        -------
            prob: 是否为人脸的概率值
            loc: 人脸位置坐标
        """
        return self.session.run([self.loc, self.prob], feed_dict={self.input: image})

class StageStatus(object):
    """
    Introduction
    ------------
        记录每个stage的预测人脸框的坐标
        其中x, y, ex, ey为相对图像大小width, height的坐标(越界截断后)
        dx, dy, edx, edy为
    """
    def __init__(self, pad_result = None, width=0, height=0):
        self.width = width
        self.height = height
        self.dy = self.edy = self.dx = self.edx = self.y = self.ey = self.x = self.ex = self.tmpw = self.tmph = []

        if pad_result is not None:
            self.update(pad_result)

    def update(self, pad_result: tuple):
        s = self
        s.dy, s.edy, s.dx, s.edx, s.y, s.ey, s.x, s.ex, s.tmpw, s.tmph = pad_result


class mtcnn:
    def __init__(self, weights_file, steps_threshold = None, min_face_size = 20, scale_factor = 0.709):
        if steps_threshold is None:
            steps_threshold = [0.6, 0.5, 0.5]
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor
        self.steps_threshold = steps_threshold
        self.graph = tf.Graph()
        config = tf.ConfigProto(log_device_placement = False)
        config.gpu_options.allow_growth = True
        with self.graph.as_default():
            self.session = tf.Session(config = config, graph = self.graph)
            weights = np.load(weights_file).item()
            self.pnet = PNet(self.session)
            self.rnet = RNet(self.session)
            self.onet = ONet(self.session)
            self.set_weigths('pnet', weights['PNet'])
            self.set_weigths('rnet', weights['RNet'])
            self.set_weigths('onet', weights['ONet'])



    def set_weigths(self, net_name, weights_value, ignore_missing = False):
        """
        Introduction
        ------------
            加载权重文件
        Parameters
        ----------
            net_name: 网络名字
            weights_value: 权重值
            ignore_missing: 忽略缺失值
        """
        with tf.variable_scope(net_name):
            for layer_name in weights_value:
                if layer_name == 'fc2-3':
                    continue
                with tf.variable_scope(layer_name, reuse = True):
                    for param_name, data in weights_value[layer_name].items():
                        try:
                            var = tf.get_variable(param_name)
                            self.session.run(var.assign(data))
                        except ValueError:
                            if not ignore_missing:
                                raise


    def compute_scale_pyramid(self, m, min_layer):
        """
        Introduction
        ------------
            生成图像金字塔缩放比例
        Parameters
        ----------
            m: 最小人脸尺寸和12比值
            min_layer: 宽度或者高度最小值和m的乘积
        Returns
        -------
            scales: 图像金字塔缩放比例列表
        """
        scales = []
        factor_count = 0

        while min_layer >= 12:
            scales += [m * np.power(self.scale_factor, factor_count)]
            min_layer = min_layer * self.scale_factor
            factor_count += 1

        return scales


    def scale_image(self, image, scale):
        """
        Introduction
        ------------
            对图像进行缩放, 归一化处理
        Parameters
        ----------
            image: 输入图像
            scale: 缩放比例
        Returns
        -------
            image_data: 缩放之后的图像
        """
        height, width, _ = image.shape

        width_scaled = int(np.ceil(width * scale))
        height_scaled = int(np.ceil(height * scale))

        im_data = cv2.resize(image, (width_scaled, height_scaled), interpolation = cv2.INTER_AREA)

        im_data_normalized = (im_data - 127.5) * 0.0078125

        return im_data_normalized

    def generate_bounding_box(self, prob, loc, scale, threshold):
        """
        Introduction
        ------------
            pnet对输入的金字塔图像全卷积生成的候选框需要进行坐标变换
        Parameters
        ----------
            prob: 每个候选框为人脸的概率值
            loc: 每个候选框人脸的坐标（相对于12*12输入图像的坐标）
            scale: 图像金字塔缩放比例
            threshold: 人脸概率阈值
        Returns
        -------
            boundingbox: 人脸候选框坐标，前四位为相对于原图像中12*12图像的坐标，最后四位为人脸相对于12*12的相对坐标
            loc: 人脸框相对于12*12的相对坐标
        """
        stride = 2
        cellsize = 12
        prob = np.transpose(prob)
        dx1 = np.transpose(loc[:, :, 0])
        dy1 = np.transpose(loc[:, :, 1])
        dx2 = np.transpose(loc[:, :, 2])
        dy2 = np.transpose(loc[:, :, 3])
        y, x = np.where(prob >= threshold)

        if y.shape[0] == 1:
            dx1 = np.flipud(dx1)
            dy1 = np.flipud(dy1)
            dx2 = np.flipud(dx2)
            dy2 = np.flipud(dy2)

        score = prob[(y, x)]
        loc = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]]))
        if loc.size == 0:
            loc = np.empty(shape = (0, 3))
        bbox = np.transpose(np.vstack([y, x]))
        q1 = np.fix((stride * bbox + 1) / scale)
        q2 = np.fix((stride * bbox + cellsize) / scale)
        boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), loc])
        return boundingbox, loc


    def nms(self, boxes, threshold):
        """
        Introduction
        ------------
            非极大值抑制
        Parameters
        ----------
            boxes: numpy box 坐标
            threshold: 非极大值抑制阈值
        """
        if boxes.size == 0:
            return np.empty((0, 3))

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        s = boxes[:, 4]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        sorted_s = np.argsort(s)

        pick = np.zeros_like(s, dtype=np.int16)
        counter = 0
        while sorted_s.size > 0:
            i = sorted_s[-1]
            pick[counter] = i
            counter += 1
            idx = sorted_s[0:-1]

            xx1 = np.maximum(x1[i], x1[idx])
            yy1 = np.maximum(y1[i], y1[idx])
            xx2 = np.minimum(x2[i], x2[idx])
            yy2 = np.minimum(y2[i], y2[idx])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            o = inter / (area[i] + area[idx] - inter)

            sorted_s = sorted_s[np.where(o <= threshold)]

        pick = pick[0:counter]

        return pick


    def rerec(self, bbox):
        """
        Introduction
        ------------
            将box坐标转换为正方形
        Parameters
        ----------
            bbox: box坐标
        """
        h = bbox[:, 3] - bbox[:, 1]
        w = bbox[:, 2] - bbox[:, 0]
        l = np.maximum(w, h)
        bbox[:, 0] = bbox[:, 0] + w * 0.5 - l * 0.5
        bbox[:, 1] = bbox[:, 1] + h * 0.5 - l * 0.5
        bbox[:, 2:4] = bbox[:, 0:2] + np.transpose(np.tile(l, (2, 1)))
        return bbox


    def pad(self, total_boxes, w, h):
        """
        Introduction
        ------------
            将box坐标超过图片边界的地方进行padding
        Parameters
        ----------
            total_boxes: 所有的人脸候选框
            w: 图片的宽度
            h: 图片的高度
        Returns
        -------
            pad截断之后的坐标和pad差值
        """
        tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
        tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
        numbox = total_boxes.shape[0]

        dx = np.ones(numbox, dtype=np.int32)
        dy = np.ones(numbox, dtype=np.int32)
        edx = tmpw.copy().astype(np.int32)
        edy = tmph.copy().astype(np.int32)

        x = total_boxes[:, 0].copy().astype(np.int32)
        y = total_boxes[:, 1].copy().astype(np.int32)
        ex = total_boxes[:, 2].copy().astype(np.int32)
        ey = total_boxes[:, 3].copy().astype(np.int32)

        tmp = np.where(ex > w)
        edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
        ex[tmp] = w

        tmp = np.where(ey > h)
        edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
        ey[tmp] = h

        tmp = np.where(x < 1)
        dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
        x[tmp] = 1

        tmp = np.where(y < 1)
        dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
        y[tmp] = 1

        return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph


    def bbreg(self, boundingbox, reg):
        """
        Introduction
        ------------
            在rnet预测的结果上修正人脸框的坐标
        Parameters
        ----------
            boundingbox: 人脸检测框
            reg: rnet预测的结果
        """
        if reg.shape[1] == 1:
            reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))
        w = boundingbox[:, 2] - boundingbox[:, 0] + 1
        h = boundingbox[:, 3] - boundingbox[:, 1] + 1
        b1 = boundingbox[:, 0] + reg[:, 0] * w
        b2 = boundingbox[:, 1] + reg[:, 1] * h
        b3 = boundingbox[:, 2] + reg[:, 2] * w
        b4 = boundingbox[:, 3] + reg[:, 3] * h
        boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
        return boundingbox


    def stage1(self, image, scales, stage_status):
        """
        Introduction
        ------------
            对输入图像进行金字塔缩放, 然后经过pnet预测人脸候选框
        Parameters
        ----------
            image: 输入图像
            scales: 缩放比例
            stage_status: 记录每个stage输出的box信息
        """
        total_boxes = np.empty(shape = (0, 9))
        status = stage_status
        for scale in scales:
            scaled_image = self.scale_image(image, scale)
            image_x = np.expand_dims(scaled_image, 0)
            #这里是因为使用的caffe训练出的模型，因此需要做翻转操作
            image_y = np.transpose(image_x, (0, 2, 1, 3))
            out = self.pnet.feed(image_y)

            out0 = np.transpose(out[0], (0, 2, 1, 3))
            out1 = np.transpose(out[1], (0, 2, 1, 3))
            boxes, _ = self.generate_bounding_box(out1[0, :, :, 1].copy(), out0[0, :, :, :].copy(), scale, self.steps_threshold[0])
            pick = self.nms(boxes.copy(), 0.5)
            if boxes.size > 0 and pick.size > 0:
                boxes = boxes[pick, :]
                total_boxes = np.append(total_boxes, boxes, axis = 0)
        num_boxes = total_boxes.shape[0]
        if num_boxes > 0:
            pick = self.nms(total_boxes.copy(), 0.7)
            total_boxes = total_boxes[pick, :]
            width = total_boxes[:, 2] - total_boxes[:, 0]
            height = total_boxes[:, 3] - total_boxes[:, 1]

            xmin = total_boxes[:, 0] + total_boxes[:, 5] * width
            ymin = total_boxes[:, 1] + total_boxes[:, 6] * height
            xmax = total_boxes[:, 2] + total_boxes[:, 7] * width
            ymax = total_boxes[:, 3] + total_boxes[:, 8] * height

            total_boxes = np.transpose(np.vstack([xmin, ymin, xmax, ymax, total_boxes[:, 4]]))
            total_boxes = self.rerec(total_boxes.copy())
            total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
            status = StageStatus(self.pad(total_boxes.copy(), stage_status.width, stage_status.height), width = stage_status.width, height = stage_status.height)
        return total_boxes, status


    def stage2(self, image, total_boxes, stage_status):
        """
        Introduction
        ------------
            对输入图像进行金字塔缩放, 然后经过rnet预测人脸候选框
        Parameters
        ----------
            image: 输入图像
            total_boxes: 经过pnet预测得到的box
            stage_status: 记录每个stage输出的box信息
        """
        num_boxes = total_boxes.shape[0]
        if num_boxes == 0:
            return total_boxes, stage_status

        #将经过pnet预测出的候选框对应的图片resize到24*24
        tempimg = np.zeros(shape=(24, 24, 3, num_boxes))

        for k in range(0, num_boxes):
            tmp = np.zeros((int(stage_status.tmph[k]), int(stage_status.tmpw[k]), 3))

            tmp[stage_status.dy[k] - 1:stage_status.edy[k], stage_status.dx[k] - 1:stage_status.edx[k], :] = image[stage_status.y[k] - 1:stage_status.ey[k], stage_status.x[k] - 1:stage_status.ex[k], :]

            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_AREA)

            else:
                return np.empty(shape=(0,)), stage_status

        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))

        out = self.rnet.feed(tempimg1)

        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])

        score = out1[1, :]

        ipass = np.where(score > self.steps_threshold[1])

        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])

        mv = out0[:, ipass[0]]

        if total_boxes.shape[0] > 0:
            pick = self.nms(total_boxes, 0.7)
            total_boxes = total_boxes[pick, :]
            total_boxes = self.bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
            total_boxes = self.rerec(total_boxes.copy())

        return total_boxes, stage_status


    def stage3(self, image, total_boxes, stage_status):
        """
        Introduction
        ------------
            对输入图像进行金字塔缩放, 然后经过rnet预测人脸候选框
        Parameters
        ----------
            image: 输入图像
            total_boxes: 经过pnet预测得到的box
            stage_status: 记录每个stage输出的box信息
        """
        num_boxes = total_boxes.shape[0]
        if num_boxes == 0:
            return total_boxes, np.empty(shape=(0,))

        total_boxes = np.fix(total_boxes).astype(np.int32)

        status = StageStatus(self.pad(total_boxes.copy(), stage_status.width, stage_status.height), width = stage_status.width, height = stage_status.height)

        tempimg = np.zeros((48, 48, 3, num_boxes))

        for k in range(0, num_boxes):

            tmp = np.zeros((int(status.tmph[k]), int(status.tmpw[k]), 3))

            tmp[status.dy[k] - 1:status.edy[k], status.dx[k] - 1:status.edx[k], :] = image[status.y[k] - 1:status.ey[k], status.x[k] - 1:status.ex[k], :]

            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_AREA)
            else:
                return np.empty(shape=(0,)), np.empty(shape=(0,))

        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))

        out = self.onet.feed(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])

        score = out1[1, :]

        ipass = np.where(score > self.steps_threshold[2])

        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])

        mv = out0[:, ipass[0]]

        if total_boxes.shape[0] > 0:
            total_boxes = self.bbreg(total_boxes.copy(), np.transpose(mv))
            pick = self.nms(total_boxes.copy(), 0.7)
            total_boxes = total_boxes[pick, :]

        return total_boxes

    def detect_face(self, image):
        """
        Introduction
        ------------
            检测图片中的人脸
        Parameters
        ----------
            image: 输入的图片
        Returns
        -------

        """
        height, width, _ = image.shape
        stage_status = StageStatus(width = width, height = height)
        minsize = round(0.05 * np.minimum(width, height))
        m = 12 / minsize
        min_layer = np.min([height, width]) * m
        scales = self.compute_scale_pyramid(m, min_layer)
        stage1_boxes, stage_1 = self.stage1(image, scales, stage_status)
        stage2_boxes, stage_2 = self.stage2(image, stage1_boxes, stage_1)
        stage3_boxes = self.stage3(image, stage2_boxes, stage_2)
        bounding_boxes = []

        for bounding_box in stage3_boxes:
            if len(bounding_box) > 0:
                bounding_boxes.append({
                    'box': [int(bounding_box[0]), int(bounding_box[1]),
                            int(bounding_box[2] - bounding_box[0]), int(bounding_box[3] - bounding_box[1])],
                    'confidence': bounding_box[-1]}
                )

        return bounding_boxes

    def __del__(self):
        self.session.close()

    def convert_pb(self):
        """
        Introduction
        ------------
            将模型导出为pb格式
        Parameters
        ----------
            output_file: 输出pb文件
        """
        output_node_names = 'pnet/conv4-2/BiasAdd,pnet/prob,rnet/fc2-2/fc2-2, rnet/prob, onet/fc2-2/fc2-2,onet/prob'
        print('Freezing graph')
        graph_def_file = './models/mtcnn-graph.pb'
        freeze_graph(input_graph = graph_def_file, input_saver="", input_binary = True, input_checkpoint = './models/model-ckpt', clear_devices = True, initializer_nodes = "", output_graph = 'frozen-mtcnn.pb', restore_op_name = 'save/restore_all', output_node_names = output_node_names, filename_tensor_name= "save/Const:0")

