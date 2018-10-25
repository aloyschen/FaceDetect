from mtcnn import mtcnn
import netron
import tfcoreml as tf_convert
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
import time
import operator
import sys

def inspect(model_pb, output_txt_file):
    graph_def = graph_pb2.GraphDef()
    with open(model_pb, "rb") as f:
        graph_def.ParseFromString(f.read())

    tf.import_graph_def(graph_def)

    sess = tf.Session()
    OPS = sess.graph.get_operations()

    ops_dict = {}

    sys.stdout = open(output_txt_file, 'w')
    for i, op in enumerate(OPS):
        print('---------------------------------------------------------------------------------------------------------------------------------------------')
        print("{}: op name = {}, op type = ( {} ), inputs = {}, outputs = {}".format(i, op.name, op.type, ", ".join([x.name for x in op.inputs]), ", ".join([x.name for x in op.outputs])))
        print('@input shapes:')
        for x in op.inputs:
            print("name = {} : {}".format(x.name, x.get_shape()))
        print('@output shapes:')
        for x in op.outputs:
            print("name = {} : {}".format(x.name, x.get_shape()))
        if op.type in ops_dict:
            ops_dict[op.type] += 1
        else:
            ops_dict[op.type] = 1

    print('---------------------------------------------------------------------------------------------------------------------------------------------')
    sorted_ops_count = sorted(ops_dict.items(), key=operator.itemgetter(1))
    print('OPS counts:')
    for i in sorted_ops_count:
        print("{} : {}".format(i[0], i[1]))

if __name__ == '__main__':
    # netron.start('./mtcnn.pb')
    # inspect('./mtcnn.pb', './all_model.txt')
    # model = mtcnn('./mtcnn_weights.npy')
    # output_node_names = ['pnet/conv4-2/BiasAdd:0', 'pnet/prob:0', 'rnet/fc2-1/fc2-1:0', 'rnet/prob:0', 'onet/prob:0', 'onet/fc2-2/fc2-2:0']
    # tf_convert.convert('./frozen-mtcnn.pb', mlmodel_path = './mtcnn.mlmodel', output_feature_names = output_node_names, input_name_shape_dict = {'pnet/input_data:0' : [1, 640, 640, 3], 'rnet/input_data:0' : [1, 24, 24, 3], 'onet/input_data:0' : [1, 48, 48, 3]})
    model = mtcnn('./mtcnn_weights.npy')
    output_node_names = ['pnet/add_2:0']
    tf_convert.convert('./mtcnn.pb', mlmodel_path = './mtcnn_all.mlmodel', output_feature_names = output_node_names, input_name_shape_dict = {'input:0' : [640, 640, 3], 'min_size:0' : [1], 'thresholds:0' : [3], 'factor:0' : [1]})
