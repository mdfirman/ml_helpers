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

def test_normalised_accuracy():
    from functools import partial

    top2acc = partial(evaluation.top_n_accuracy, n=2)

    ground_truth = [0, 1, 2, 2]
    preds = np.array([[0.2, 0.8, 0.0],
                      [0.0, 0.6, 0.4],
                      [0.0, 0.0, 1.0],
                      [0.9, 0.0, 0.1]])

    acc = evaluation.class_normalised_accuracy(
        ground_truth, preds, accuracy_function=top2acc)
    assert acc == 1.0

    acc = evaluation.class_normalised_accuracy(ground_truth, preds)
    assert acc == 0.5


test_top_n_evaluation()
test_normalised_accuracy()
