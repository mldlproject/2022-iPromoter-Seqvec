# Import libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
import os

#====================================================================================#
def performance(labels, probs, threshold=0.5, dataset='test_set', printout=False, csv_export=False, tag='', group='', decimal=4):
    d = decimal
    _threshold = threshold
    _probs = probs
    _dataset = dataset
    #-------------------------
    if _threshold != 0.5:
        predicted_labels = []
        for prob in _probs:
            if prob >= _threshold:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
    else:
        predicted_labels = np.round(_probs)
    #-------------------------
    tn, fp, fn, tp   = confusion_matrix(labels, predicted_labels).ravel()
    auc_roc          = np.round(roc_auc_score(labels, probs), d)
    auc_pr           = np.round(average_precision_score(labels, probs), d)
    acc              = np.round(accuracy_score(labels, predicted_labels), d)
    ba               = np.round(balanced_accuracy_score(labels, predicted_labels), d)
    sensitivity      = np.round(tp / (tp + fn), d)
    specificity      = np.round(tn / (tn + fp), d)
    precision        = np.round(tp / (tp + fp), d)
    mcc              = np.round(matthews_corrcoef(labels, predicted_labels), d)
    f1               = np.round(2*precision*sensitivity / (precision + sensitivity), d)
    ck               = np.round(cohen_kappa_score(labels, predicted_labels), d)
    #-------------------------
    if printout:
        print('Performance of {}'.format(_dataset))
        print('AUC-ROC: {}, AUC-PR: {}, ACC: {}, BA : {}, SN: {}, SP: {}, PR: {}, MCC: {}, F1: {}, CK {}'.format(auc_roc, auc_pr, acc, ba, sensitivity, specificity, precision, mcc, f1, ck))
    #-------------------------
    if csv_export:
        directory = './performance/{}'.format(group)
        if os.path.isdir(directory) == False:
            os.makedirs(directory)
        metrics = ['AUC-ROC', 'AUC-PR', 'ACC', 'BA', 'SN', 'SP', 'PR', 'MCC', 'F1', 'CK']
        values  = [auc_roc, auc_pr, acc, ba, sensitivity, specificity, precision, mcc, f1, ck]       
        pd.DataFrame([values], columns=metrics).to_csv("./performance/{}/{}_performace.csv".format(group, tag), index=False)
    #-------------------------    
    return auc_roc, auc_pr, acc, ba, sensitivity, specificity, precision, mcc, f1, ck