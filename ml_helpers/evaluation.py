import numpy as np
from sklearn import metrics


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
    Accuracy done per-class and averaged across all classes.
    This mitigates against the effect of large class imbalance in the test
    set. However, it can make evaluation sensistive to changes in
    classification on very infrequent classes - you can use min_class_count
    to mitigate against this problem
    Cite: brostow
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
    Plots a single roc curve with a dot
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
