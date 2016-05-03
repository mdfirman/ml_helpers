import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


def multiclass_auc(gt, preds):
    """
    Compute a class-weighted multiclass AUC, as defined in section 9.2 of [1].

    Parameters
    ----------
    gt : sequence
        A 1D vector of ground truth class labels
    preds : array
        An array of classifier predictions, where columns correspond to classes
        and rows to data instances.

    [1] https://ccrma.stanford.edu/workshops/mir2009/references/ROCintro.pdf
    """

    assert gt.shape[0] == preds.shape[0]
    assert gt.max() < preds.shape[1]

    # compute AUC for each class in turn
    aucs = []
    weights = []
    for class_id in range(preds.shape[1]):
        class_gt = gt == class_id
        if class_gt.sum() == 0:
            continue
        class_preds = preds[:, class_id]
        aucs.append(metrics.roc_auc_score(class_gt, class_preds))
        weights.append((gt == class_id).sum())

    # return the class-weighted mean of the AUCs
    return np.average(np.array(aucs), weights=np.array(weights))


def class_normalised_accuracy_score(gt, pred, min_class_count=1):
    """
    Compute class-normalised classification accuracy score

    Accuracy done per-class and averaged across all classes.
    This mitigates against the effect of large class imbalance in the test
    set (see e.g. [1]). However, it can make evaluation sensistive to changes
    in classification on very infrequent classes - you can use min_class_count
    to deal with this.

    Parameters
    ----------
    gt : array
        Integers representing ground truth class labels
    pred : array
        Integers representing predicted labels
    min_class_count : integer, optional
        Ground truth classes with fewer than this number of items in them
        are ignored in the final computation

    [1] Gabriel J. Brostow, Jamie Shotton, Julien Fauqueur and Roberto Cipolla
    Segmentation and Recognition using Structure from Motion Point Clouds
    """
    assert gt.shape == pred.shape

    accs = []
    for target in np.unique(gt):
        idxs = gt == target

        # only include if we have enough ground truth classes
        if idxs.sum() >= min_class_count:
            accs.append(metrics.accuracy_score(gt[idxs], pred[idxs]))

    return np.mean(accs)


def plot_roc_curve(gt, pred, label="", plot_midpoint=True):
    """
    Plot a single roc curve, optionally with a dot at the 0.5 threshold
    """

    # evaluating and finding the curve midpoint
    fpr, tpr, thresh = metrics.roc_curve(gt, pred.ravel())
    roc_auc = metrics.auc(fpr, tpr)

    # plotting curve
    plt.plot(fpr, tpr , label='%s (area = %0.2f)' % (label, roc_auc))

    if plot_midpoint:
        mid_idx = np.argmin(np.abs(thresh-0.5))
        plt.plot(fpr[mid_idx], tpr[mid_idx], 'bo')

    # labels and legends
    plt.legend(loc='best')
    plt.xlabel('FPR')
    plt.ylabel('TPR')


def top_n_accuracy(gt, pred, n):
    """
    Computes the fraction of items which have the correct answer in the top
    n predictions made by a classifier.

    Parameters
    ----------
    gt : array
        Integers representing ground truth class labels
    pred : array
        Matrix representing class probabilities as predicted by a classifer
    n : integer
        If the ground truth is in the top n prediction of the classifier,
        the item is counted as a successful prediction
    """
    sorted_preds = np.argsort(pred, axis=1)[:, ::-1][:, :n]

    successes = 0
    for pred_classes, ground_truth_class in zip(sorted_preds, gt):
        if ground_truth_class in pred_classes:
            successes += 1

    return float(successes) / len(gt)
