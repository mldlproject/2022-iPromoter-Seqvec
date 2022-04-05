# Import libraries
import os
import csv
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score

#====================================================================================#
def save_result(dictionary, save_dir="/content/drive/My Drive/Predict_task/result", filename='Result.csv'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = os.path.join(save_dir, filename)
    if not (os.path.exists(path)):
        logfile   = open(path, 'a')
        logwriter = csv.DictWriter(logfile, fieldnames=list(dictionary.keys()))
        logwriter.writeheader()
        logwriter = csv.DictWriter(logfile, fieldnames = dictionary.keys())
        logwriter.writerow(dictionary)
    else:
        logfile   = open(path, 'a')
        logwriter = csv.DictWriter(logfile, fieldnames=dictionary.keys())
        logwriter.writerow(dictionary)
    logfile.close()

#====================================================================================#
# Get probabilities 
def get_prob(prob_list, best_epoch):
    bestE_problist = []
    for batch_y in prob_list[best_epoch]:
        for i in batch_y:
            bestE_problist.append(i.detach().cpu().numpy())
    bestE_problist = np.array(bestE_problist)
    return bestE_problist

#====================================================================================#
# Get model performance
def performance(labels, probs, threshold=0.5, name='test_dataset', printout=False, path_save = None, decimal=4):
    d = decimal
    _threshold = threshold
    _probs = probs
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
    tn, fp, fn, tp  = confusion_matrix(labels, predicted_labels).ravel()
    auc_roc         = np.round(roc_auc_score(labels, probs), d)
    auc_pr          = np.round(average_precision_score(labels, probs), d)
    acc             = np.round(accuracy_score(labels, predicted_labels), d)
    ba              = np.round(balanced_accuracy_score(labels, predicted_labels), d)
    sensitivity     = np.round(tp / (tp + fn), d)
    specificity     = np.round(tn / (tn + fp), d)
    precision       = np.round(tp / (tp + fp), d)
    mcc             = np.round(matthews_corrcoef(labels, predicted_labels), d)
    f1              = np.round(2*precision*sensitivity / (precision + sensitivity), d)
    ck              = np.round(cohen_kappa_score(labels, predicted_labels), d)
    #-------------------------
    if printout:
        print('Performance for {}'.format(name))
        print('AUC-ROC: {}, AUC-PR: {}, ACC: {}, BA : {}, SN: {}, SP: {}, PR: {}, MCC: {}, F1: {}, CK {}'.format(auc_roc, auc_pr, acc, ba, sensitivity, specificity, precision, mcc, f1, ck))
    #-------------------------
    result = {}
    result['Dataset'] = name
    result['AUC-ROC'] = auc_roc
    result['AUC-PR']  = auc_pr
    result['ACC']     = acc
    result['BA']      = ba
    result['SN']      = sensitivity
    result['SP']      = specificity
    result['PR']      = precision
    result['MCC']     = mcc
    result['CK']      = ck
    #-------------------------
    if path_save:
        save_result(result, save_dir= path_save, filename='result_test_cut.csv')
    #-------------------------
    return auc_roc, auc_pr, acc, ba, sensitivity, specificity, precision, mcc, f1, ck
