import sys
import numpy as np
sys.path.append('../')
import evaluation

def test_top_n_evaluation():
    preds = np.array([[10, 2, 3, 5, 6], [8, 0, 0, 2, 10]]).astype(np.float32)
    preds /= preds.sum(1)[:, None]
    gt = [0, 0]
    assert evaluation.top_n_accuracy(gt, preds, 1) == 0.5
    assert evaluation.top_n_accuracy(gt, preds, 2) == 1.0
    assert evaluation.top_n_accuracy([1, 1], preds, 2) == 0.0


test_top_n_evaluation()
