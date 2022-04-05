import numpy as np
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix, roc_auc_score, average_precision_score, cohen_kappa_score, balanced_accuracy_score

def printPerformance(labels, probs, printout=False, decimal=4):
    predicted_labels = np.round(probs)
    tn, fp, fn, tp = confusion_matrix(labels, predicted_labels).ravel()
    acc = accuracy_score(labels, predicted_labels)
    ba  = balanced_accuracy_score(labels, predicted_labels)
    roc_auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    mcc = matthews_corrcoef(labels, predicted_labels)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision   = tp / (tp + fp)
    f1 = 2*precision*sensitivity / (precision + sensitivity)
    ck = cohen_kappa_score(labels, predicted_labels)
    d = decimal
    if printout:
        print('AUC-ROC: ', round(roc_auc, d))
        print('AUC-PR: ', round(pr_auc, d))
        print('ACC: ', round(acc, d))
        print('BA: ', round(acc, d))
        print('SN/RE: ', round(sensitivity, d))
        print('SP: ', round(specificity, d))
        print('PR: ', round(precision, d))
        print('MCC: ', round(mcc, d))
        print('F1: ', round(f1, d))
    return round(roc_auc, d), round(pr_auc, d), round(acc, d), round(ba, d), round(sensitivity, 4), round(specificity, 4), round(precision, 4), round(mcc, 4), round(f1, 4), round(ck, 4) 