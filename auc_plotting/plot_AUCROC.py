import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, roc_auc_score
import os

#--------------------------------------------------
# Define data information
iPro_EL_hs_TA_prob = pd.read_csv("./iPro_EL/iPro_EL_hs_TA_test.csv")['Proba_R']
iPro_Seqvec_hs_TA_prob = np.load("./iPromoter-Seqvec/probs_hs_TA.npy")

#--------------------------------------------------
model_probs_hs_TA = [iPro_EL_hs_TA_prob, iPro_Seqvec_hs_TA_prob]
model_name_hs_TA = ['iPro-EL', 'iPromoter-Seqvec']    

# AUCROC plot
fpr_list, tpr_list, auc_list = [], [], []
for i in range(len(model_probs_hs_TA)):
    labels_hs_TA = np.load("./iPromoter-Seqvec/label_hs_TA.npy")
    prob = model_probs_hs_TA[i]
    fpr, tpr, _ = roc_curve(labels_hs_TA,  prob)
    auc = np.round(roc_auc_score(labels_hs_TA, prob), 2)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    auc_list.append(auc)

fig = plt.figure(figsize=(8,8))
for i in range(2):
    plt.plot(fpr_list[i], tpr_list[i], label="{}, AUC={:.2f}".format(model_name_hs_TA[i], auc_list[i]))
    plt.plot([0,1], [0,1], color='red', linestyle='--')
    
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('ROC curves of iPromoter-Seqvec and iPro-EL (HS-TAPro)', fontsize=15)
plt.legend(prop={'size':10}, loc='lower right')

PATH = './images'
if os.path.isdir(PATH):
    fig.savefig('{}/AUCROC_{}.pdf'.format(PATH, "hs_TA"))
else:
    os.makedirs(PATH)
    fig.savefig('{}/AUCROC_{}.pdf'.format(PATH, "hs_TA"))

plt.show()

#--------------------------------------------------
# Define data information
iPro_EL_hs_nonTA_prob = pd.read_csv("./iPro_EL/iPro_EL_hs_nonTA_test.csv")['Proba_R']
iPro_Seqvec_hs_nonTA_prob = np.load("./iPromoter-Seqvec/probs_hs_nonTA.npy")

#--------------------------------------------------
model_probs_hs_nonTA = [iPro_EL_hs_nonTA_prob, iPro_Seqvec_hs_nonTA_prob]
model_name_hs_nonTA = ['iPro-EL', 'iPromoter-Seqvec']    

# AUCROC plot
fpr_list, tpr_list, auc_list = [], [], []
for i in range(len(model_probs_hs_nonTA)):
    labels_hs_TA = np.load("./iPromoter-Seqvec/label_hs_nonTA.npy")
    prob = model_probs_hs_nonTA[i]
    fpr, tpr, _ = roc_curve(labels_hs_TA,  prob)
    auc = np.round(roc_auc_score(labels_hs_TA, prob), 2)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    auc_list.append(auc)

fig = plt.figure(figsize=(8,8))
for i in range(2):
    plt.plot(fpr_list[i], tpr_list[i], label="{}, AUC={:.2f}".format(model_name_hs_nonTA[i], auc_list[i]))
    plt.plot([0,1], [0,1], color='red', linestyle='--')
    
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('ROC curves of iPromoter-Seqvec and iPro-EL (HS-nonTAPro)', fontsize=15)
plt.legend(prop={'size':10}, loc='lower right')

PATH = './images'
if os.path.isdir(PATH):
    fig.savefig('{}/AUCROC_{}.pdf'.format(PATH, "hs_nonTA"))
else:
    os.makedirs(PATH)
    fig.savefig('{}/AUCROC_{}.pdf'.format(PATH, "hs_nonTA"))

plt.show()

#--------------------------------------------------
# Define data information
iPro_Seqvec_mm_TA_prob = np.load("./iPromoter-Seqvec/probs_mm_TA.npy")

# AUCROC plot
labels_mm_TA = np.load("./iPromoter-Seqvec/label_mm_TA.npy")
probs_mm_TA = iPro_Seqvec_mm_TA_prob 
fpr, tpr, _ = roc_curve(labels_mm_TA,  probs_mm_TA)
auc = np.round(roc_auc_score(labels_mm_TA, probs_mm_TA), 2)

fig = plt.figure(figsize=(8,8))
plt.plot(fpr, tpr, label="{}, AUC={:.2f}".format('iPromoter-Seqvec', auc))
plt.plot([0,1], [0,1], color='red', linestyle='--')
    
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('ROC curve of iPromoter-Seqvec (MM-TAPro)', fontsize=15)
plt.legend(prop={'size':10}, loc='lower right')

PATH = './images'
if os.path.isdir(PATH):
    fig.savefig('{}/AUCROC_{}.pdf'.format(PATH, "mm_TA"))
else:
    os.makedirs(PATH)
    fig.savefig('{}/AUCROC_{}.pdf'.format(PATH, "mm_TA"))

plt.show()

#--------------------------------------------------
# Define data information
iPro_Seqvec_mm_nonTA_prob = np.load("./iPromoter-Seqvec/probs_mm_nonTA.npy")

# AUCROC plot
labels_mm_TA = np.load("./iPromoter-Seqvec/label_mm_nonTA.npy")
prob_mm_TA = iPro_Seqvec_mm_nonTA_prob
fpr, tpr, _ = roc_curve(labels_mm_TA,  prob_mm_TA)
auc = np.round(roc_auc_score(labels_mm_TA, prob_mm_TA), 2)

fig = plt.figure(figsize=(8,8))
plt.plot(fpr, tpr, label="{}, AUC={:.2f}".format('iPromoter-Seqvec', auc))
plt.plot([0,1], [0,1], color='red', linestyle='--')
    
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('ROC curve of iPromoter-Seqvec (MM-nonTAPro)', fontsize=15)
plt.legend(prop={'size':10}, loc='lower right')

PATH = './images'
if os.path.isdir(PATH):
    fig.savefig('{}/AUCROC_{}.pdf'.format(PATH, "mm_nonTA"))
else:
    os.makedirs(PATH)
    fig.savefig('{}/AUCROC_{}.pdf'.format(PATH, "mm_nonTA"))

plt.show()

