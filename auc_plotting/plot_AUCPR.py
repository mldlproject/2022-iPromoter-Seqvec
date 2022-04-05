import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import precision_recall_curve, average_precision_score
import os

#--------------------------------------------------
# Define data information
iPro_EL_hs_TA_prob = pd.read_csv("./iPro_EL/iPro_EL_hs_TA_test.csv")['Proba_R']
iPro_Seqvec_hs_TA_prob = np.load("./iPromoter-Seqvec/probs_hs_TA.npy")

#--------------------------------------------------
model_probs_hs_TA = [iPro_EL_hs_TA_prob, iPro_Seqvec_hs_TA_prob]
model_name_hs_TA = ['iPro-EL', 'iPromoter-Seqvec']    

# AUCROC plot
precision_list, recall_list, auc_list = [], [], []
for i in range(len(model_probs_hs_TA)):
    labels_hs_TA = np.load("./iPromoter-Seqvec/label_hs_TA.npy")
    prob = model_probs_hs_TA[i] 
    precision, recall, _ = precision_recall_curve(labels_hs_TA,  prob)
    auc = np.round(average_precision_score(labels_hs_TA, prob), 2)
    precision_list.append(precision)
    recall_list.append(recall)
    auc_list.append(auc)


fig = plt.figure(figsize=(8,8))
for i in range(2):
    plt.plot(recall_list[i], precision_list[i], label="{}, AUC={:.2f}".format(model_name_hs_TA[i], auc_list[i]))
    #plt.plot([1,0], [0,1], color='blue', linestyle='--')
    
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('PR curves of iPromoter-Seqvec and iPro-EL (HS-TAPro)', fontsize=15)
plt.legend(prop={'size':10}, loc='lower right')

PATH = './images'
if os.path.isdir(PATH):
    fig.savefig('{}/AUCPR_{}.pdf'.format(PATH, "hs_TA"))
else:
    os.makedirs(PATH)
    fig.savefig('{}/AUCPR_{}.pdf'.format(PATH, "hs_TA"))

plt.show()

#--------------------------------------------------
# Define data information
iPro_EL_hs_nonTA_prob = pd.read_csv("./iPro_EL/iPro_EL_hs_nonTA_test.csv")['Proba_R']
iPro_Seqvec_hs_nonTA_prob = np.load("./iPromoter-Seqvec/probs_hs_nonTA.npy")

#--------------------------------------------------
model_probs_hs_nonTA = [iPro_EL_hs_nonTA_prob, iPro_Seqvec_hs_nonTA_prob]
model_name_hs_nonTA = ['iPro-EL', 'iPromoter-Seqvec']    

# AUCROC plot
precision_list, recall_list, auc_list = [], [], []
for i in range(len(model_probs_hs_nonTA)):
    labels_hs_NONTA = np.load("./iPromoter-Seqvec/label_hs_nonTA.npy")
    prob = model_probs_hs_nonTA[i] 
    precision, recall, _ = precision_recall_curve(labels_hs_NONTA,  prob)
    auc = np.round(average_precision_score(labels_hs_NONTA, prob), 2)
    precision_list.append(precision)
    recall_list.append(recall)
    auc_list.append(auc)

fig = plt.figure(figsize=(8,8))
for i in range(2):
    plt.plot(recall_list[i], precision_list[i], label="{}, AUCPR={:.2f}".format(model_name_hs_nonTA[i], auc_list[i]))
    #plt.plot([1,0], [0,1], color='blue', linestyle='--')
    
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('PR curves of iPromoter-Seqvec and iPro-EL (HS-nonTAPro)', fontsize=15)
plt.legend(prop={'size':10}, loc='lower right')

PATH = './images'
if os.path.isdir(PATH):
    fig.savefig('{}/AUCPR_{}.pdf'.format(PATH, "hs_nonTA"))
else:
    os.makedirs(PATH)
    fig.savefig('{}/AUCPR_{}.pdf'.format(PATH, "hs_nonTA"))

plt.show()

#--------------------------------------------------
# Define data information
iPro_Seqvec_mm_TA_prob = np.load("./iPromoter-Seqvec/probs_mm_TA.npy")

# AUCROC plot
labels_mm_TA = np.load("./iPromoter-Seqvec/label_mm_TA.npy")
iPro_Seqvec_mm_TA_prob = np.load("./iPromoter-Seqvec/probs_mm_TA.npy")
precision, recall, _ = precision_recall_curve(labels_mm_TA,  iPro_Seqvec_mm_TA_prob)
auc = np.round(average_precision_score(labels_mm_TA, iPro_Seqvec_mm_TA_prob), 2)


fig = plt.figure(figsize=(8,8))
plt.plot(recall, precision, label="{}, AUCPR={:.2f}".format('iPromoter-Seqvec', auc))
    #plt.plot([1,0], [0,1], color='blue', linestyle='--')
    
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('PR curves of iPromoter-Seqvec (MM-TAPro)', fontsize=15)
plt.legend(prop={'size':10}, loc='lower right')

PATH = './images'
if os.path.isdir(PATH):
    fig.savefig('{}/AUCPR_{}.pdf'.format(PATH, "mm_TA"))
else:
    os.makedirs(PATH)
    fig.savefig('{}/AUCPR_{}.pdf'.format(PATH, "mm_TA"))

plt.show()

#--------------------------------------------------
# Define data information
iPro_Seqvec_mm_nonTA_prob = np.load("./iPromoter-Seqvec/probs_mm_nonTA.npy")

# AUCROC plot
labels_mm_nonTA = np.load("./iPromoter-Seqvec/label_mm_nonTA.npy")
iPro_Seqvec_mm_TA_prob = np.load("./iPromoter-Seqvec/probs_mm_nonTA.npy")
precision, recall, _ = precision_recall_curve(labels_mm_nonTA,  iPro_Seqvec_mm_nonTA_prob)
auc = np.round(average_precision_score(labels_mm_nonTA, iPro_Seqvec_mm_nonTA_prob), 2)


fig = plt.figure(figsize=(8,8))
plt.plot(recall, precision, label="{}, AUCPR={:.2f}".format('iPromoter-Seqvec', auc))
    #plt.plot([1,0], [0,1], color='blue', linestyle='--')
    
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('PR curves of iPromoter-Seqvec (MM-nonTAPro)', fontsize=15)
plt.legend(prop={'size':10}, loc='lower right')

PATH = './images'
if os.path.isdir(PATH):
    fig.savefig('{}/AUCPR_{}.pdf'.format(PATH, "mm_nonTA"))
else:
    os.makedirs(PATH)
    fig.savefig('{}/AUCPR_{}.pdf'.format(PATH, "mm_nonTA"))

plt.show()