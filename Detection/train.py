import os
import numpy as np
import tensorflow as tf
from datasets import data as dataset
from models.nn import YOLO as ConvNet
from learning.optimizers import MomentumOptimizer as Optimizer
# from learning.optimizers import AdamOptimizer as Optimizer
from learning.evaluators import RecallEvaluator as Evaluator

""" 1. 원본 데이터셋을 메모리에 로드하고 분리함 """
root_dir = os.path.join('data/face/') # FIXME
trainval_dir = os.path.join(root_dir, 'train')

# 앵커 로드
anchors = dataset.load_json(os.path.join(trainval_dir, 'anchors.json'))

# 학습에 사용될 이미지 사이즈 및 클래스 개수를 정함
IM_SIZE = (416, 416)
NUM_CLASSES = 3

# 원본 학습+검증 데이터셋을 로드하고, 이를 학습 데이터셋과 검증 데이터셋으로 나눔
X_trainval, y_trainval = dataset.read_data(trainval_dir, IM_SIZE)
trainval_size = X_trainval.shape[0]
val_size = int(trainval_size * 0.1) # FIXME
val_set = dataset.DataSet(X_trainval[:val_size], y_trainval[:val_size])
train_set = dataset.DataSet(X_trainval[val_size:], y_trainval[val_size:])

""" 2. 학습 수행 및 성능 평가를 위한 하이퍼파라미터 설정"""
hp_d = dict()

# FIXME: 학습 관련 하이퍼파라미터
hp_d['batch_size'] = 4
hp_d['num_epochs'] = 50
hp_d['init_learning_rate'] = 1e-4
hp_d['momentum'] = 0.9
hp_d['learning_rate_patience'] = 10
hp_d['learning_rate_decay'] = 0.1
hp_d['eps'] = 1e-8
hp_d['score_threshold'] = 1e-4
hp_d['nms_flag'] = True

""" 3. Graph 생성, session 초기화 및 학습 시작 """
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = ConvNet([IM_SIZE[0], IM_SIZE[1], 3], NUM_CLASSES, anchors, grid_size=(IM_SIZE[0]//32, IM_SIZE[1]//32))

evaluator = Evaluator()
optimizer = Optimizer(model, train_set, evaluator, val_set=val_set, **hp_d)

sess = tf.Session(graph=graph, config=config)
train_results = optimizer.train(sess, details=True, verbose=True, **hp_d)