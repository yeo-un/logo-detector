import os
import numpy as np
import tensorflow as tf
from datasets import data as dataset
from models.nn import YOLO as ConvNet
from learning.evaluators import RecallEvaluator as Evaluator
from learning.utils import draw_pred_boxes, predict_nms_boxes, convert_boxes
import cv2
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

""" 1. 원본 데이터셋을 메모리에 로드함 """
root_dir = os.path.join('data/face')
test_dir = os.path.join(root_dir, 'test')
IM_SIZE = (416, 416)
NUM_CLASSES = 3

# 테스트 데이터셋을 로드함
X_test, y_test = dataset.read_data(test_dir, IM_SIZE)
test_set = dataset.DataSet(X_test, y_test)

""" 2. 테스트를 위한 하이퍼파라미터 설정 """
anchors = dataset.load_json(os.path.join(test_dir, 'anchors.json'))
class_map = dataset.load_json(os.path.join(test_dir, 'classes.json'))
nms_flag = True
hp_d = dict()

# FIXME
hp_d['batch_size'] = 8
hp_d['nms_flag'] = nms_flag

""" 3. Graph 생성, 파라미터 로드, session 초기화 및 테스트 시작 """
# 초기화
graph = tf.compat.v1.get_default_graph()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([IM_SIZE[0], IM_SIZE[1], 3], NUM_CLASSES, anchors, grid_size=(IM_SIZE[0]//32, IM_SIZE[1]//32))
evaluator = Evaluator()
saver = tf.compat.v1.train.Saver()

a = './tmp/model.ckpt'

sess = tf.compat.v1.Session(graph=graph, config=config)
saver.restore(sess, a)
test_y_pred = model.predict(sess, test_set, **hp_d)
test_score = evaluator.score(test_set.labels, test_y_pred)

print('Test performance: {}'.format(test_score))

""" 4. 이미지에 바운딩 박스 그리기 시작 """
draw_dir = os.path.join(test_dir, 'redraws') # FIXME
im_dir = os.path.join(test_dir, 'images') # FIXME
im_paths = []
im_paths.extend(glob.glob('./data/face/test/images/*.jpg'))

for idx, (img, y_pred, im_path) in enumerate(zip(test_set.images, test_y_pred, im_paths)):
    name = im_path.split('/')[-1]
    draw_path =os.path.join(draw_dir, name)
    #print(draw_path)
    if nms_flag:
        bboxes = predict_nms_boxes(y_pred, conf_thres=0.5, iou_thres=0.5)
    else:
        bboxes = convert_boxes(y_pred)
    bboxes = bboxes[np.nonzero(np.any(bboxes > 0, axis=1))]
    boxed_img = draw_pred_boxes(img, bboxes, class_map)
    cv2.imwrite(draw_path, boxed_img)
	
