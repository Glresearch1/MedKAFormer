from collections import namedtuple

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


Metrics = namedtuple("Metrics", ["AUC", "ACC"])


def evaluate(y_score, labels):
    assert y_score.shape[0] == labels.shape[0]

    auc = get_auc(labels, y_score)
    acc = get_acc(labels, y_score)
    return Metrics(auc, acc)


def get_auc(y_true, y_score):
    y_true = np.asarray(y_true).squeeze()
    y_score = np.asarray(y_score).squeeze()

    auc_scores = []
    for class_idx in range(y_score.shape[1]):
        y_true_binary = (y_true == class_idx).astype(float)
        if len(np.unique(y_true_binary)) < 2:
            continue
        auc_scores.append(roc_auc_score(y_true_binary, y_score[:, class_idx]))

    if not auc_scores:
        return float("nan")
    return float(np.mean(auc_scores))


def get_acc(y_true, y_score):
    y_true = np.asarray(y_true).squeeze()
    y_score = np.asarray(y_score).squeeze()
    return float(accuracy_score(y_true, np.argmax(y_score, axis=-1)))


# Backward-compatible aliases for older experiment scripts.
def getAUC(y_true, y_score):
    return get_auc(y_true, y_score)


def getACC(y_true, y_score, threshold=0.5):
    return get_acc(y_true, y_score)
