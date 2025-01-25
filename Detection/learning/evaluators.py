from abc import abstractmethod, abstractproperty, ABCMeta
import numpy as np
from learning.utils import convert_boxes, predict_nms_boxes, cal_recall

class Evaluator(metaclass=ABCMeta):
    """성능 평가를 위한 evaluator의 베이스 클래스."""

    @abstractproperty
    def worst_score(self):
        """
        최저 성능 점수.
        :return float.
        """
        pass

    @abstractproperty
    def mode(self):
        """
        점수가 높아야 성능이 우수한지, 낮아야 성능이 우수한지 여부. 'max'와 'min' 중 하나.
        e.g. 정확도, AUC, 정밀도, 재현율 등의 경우 'max',
             오류율, 미검률, 오검률 등의 경우 'min'.
        :return: str.
        """
        pass

    @abstractmethod
    def score(self, y_true, y_pred):
        """
        실제로 사용할 성능 평가 지표.
        해당 함수를 추후 구현해야 함.
        :param y_true: np.ndarray, shape: (N, num_classes).
        :param y_pred: np.ndarray, shape: (N, num_classes).
        :return float.
        """
        pass

    @abstractmethod
    def is_better(self, curr, best, **kwargs):
        """
        현재 주어진 성능 점수가 현재까지의 최고 성능 점수보다 우수한지 여부를 반환하는 함수.
        해당 함수를 추후 구현해야 함.
        :param curr: float, 평가 대상이 되는 현재 성능 점수.
        :param best: float, 현재까지의 최고 성능 점수.
        :return bool.
        """
        pass

class RecallEvaluator(Evaluator):
    """ 재현율(Recall)을 성능 평가 청도로 사용하는 evaluator 클래스"""

    @property
    def worst_score(self):
        """최저 성능 점수"""
        return 0.0

    @property
    def mode(self):
        """점수가 높아야 성능이 우수한지 낮아야 우수한지 여부"""
        return 'max'

    def score(self, y_true, y_pred, **kwargs):
        """
        주어진 바운딩 박스에 대한 Recall 성능 평가 점수
        :param kwargs: dict, 추가 인자.
            - nms_flag: bool, True면 nms 수행
        """
        nms_flag = kwargs.pop('nms_flag', True)
        if nms_flag:
            bboxes = predict_nms_boxes(y_pred)
        else:
            bboxes = convert_boxes(y_pred)
        gt_bboxes = convert_boxes(y_true)
        score = cal_recall(gt_bboxes, bboxes)
        return score

    def is_better(self, curr, best, **kwargs):
        """
        상대적 문턱값을 고려하여, 현재 주어진 성능 점수가 현재까지의 최고 성능 점수보다
        우수한지 여부를 반환하는 함수.
        :param kwargs: dict, 추가 인자.
            - score_threshold: float, 새로운 최적값 결정을 위한 상대적 문턱값으로,
                               유의미한 차이가 발생했을 경우만을 반영하기 위함.
        """
        score_threshold = kwargs.pop('score_threshold', 1e-4)
        relative_eps = 1.0 + score_threshold
        return curr > best * relative_eps


class IoUEvaluator(Evaluator):
    """Evaluator with IoU(graph) metric."""

    @property
    def worst_score(self):
        """The worst performance score."""
        return 0.0

    @property
    def mode(self):
        """The mode for performance score."""
        return 'max'

    def score(self, sess, model, X, y):
        """
        Compute iou for a given prediction using YOLO model.
        :param sess: tf.Session.
        :param X: np.ndarray, sample image batches
        :param y: np.ndarray, sample labels batches
        :return iou: float. intersection of union
        """
        iou = sess.run(model.iou, feed_dict={model.X: X, model.y: y, model.is_train: False})
        score = np.mean(iou)
        return score

    def is_better(self, curr, best, **kwargs):
        """
        Return whether current performance scores is better than current best,
        with consideration of the relative threshold to the given performance score.
        :param kwargs: dict, extra arguments.
            - score_threshold: float, relative threshold for measuring the new optimum,
                               to only focus on significant changes.
        """
        score_threshold = kwargs.pop('score_threshold', 1e-4)
        relative_eps = 1.0 + score_threshold
        return curr > best * relative_eps