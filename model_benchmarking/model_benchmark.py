# Import libraries
from utils import *
import pandas as pd
import numpy as np

#====================================================================================#
TA_label    = [1]*250  + [0]*250
nonTA_label = [1]*2500 + [0]*2500

#====================================================================================#
# 1. Benchmarking with DeePromoter
# HS TA
hs_TA_DeePromoter= pd.read_csv("./DeePromoter/DeePromoter_hs_TA_test.csv")
predicted_hs_TA_DeePromoter = []
for i in range(len(hs_TA_DeePromoter)):
    if hs_TA_DeePromoter['Result'][i][0] == 'P':
        predicted_hs_TA_DeePromoter.append(1)
    else:
        predicted_hs_TA_DeePromoter.append(0)
printPerformance(TA_label, predicted_hs_TA_DeePromoter, printout=True, csv_export=True, tag='hs_TA', group='DeePromoter', decimal=2)

# HS nonTA
hs_nonTA_DeePromoter= pd.read_csv("./DeePromoter/DeePromoter_hs_nonTA_test.csv")
predicted_hs_nonTA_DeePromoter = []
for i in range(len(hs_nonTA_DeePromoter)):
    if hs_nonTA_DeePromoter['Result'][i][0] == 'P':
        predicted_hs_nonTA_DeePromoter.append(1)
    else:
        predicted_hs_nonTA_DeePromoter.append(0)
printPerformance(nonTA_label, predicted_hs_nonTA_DeePromoter, printout=True, csv_export=True, tag='hs_nonTA', group='DeePromoter', decimal=2)

# MM TA
mm_TA_DeePromoter= pd.read_csv("./DeePromoter/DeePromoter_mm_TA_test.csv")
predicted_mm_TA_DeePromoter = []
for i in range(len(mm_TA_DeePromoter)):
    if mm_TA_DeePromoter['Result'][i][0] == 'P':
        predicted_mm_TA_DeePromoter.append(1)
    else:
        predicted_mm_TA_DeePromoter.append(0)
printPerformance(TA_label, predicted_mm_TA_DeePromoter, printout=True, csv_export=True, tag='mm_TA', group='DeePromoter', decimal=2)

# MM nonTA
mm_nonTA_DeePromoter= pd.read_csv("./DeePromoter/DeePromoter_mm_nonTA_test.csv")
predicted_mm_nonTA_DeePromoter = []
for i in range(len(mm_nonTA_DeePromoter)):
    if mm_nonTA_DeePromoter['Result'][i][0] == 'P':
        predicted_mm_nonTA_DeePromoter.append(1)
    else:
        predicted_mm_nonTA_DeePromoter.append(0)
printPerformance(nonTA_label, predicted_mm_nonTA_DeePromoter, printout=True, csv_export=True, tag='mm_nonTA', group='DeePromoter', decimal=2)

#====================================================================================#
# 1. Benchmarking with iPro-EL
# HS TA
hs_TA_iPro_EL= pd.read_csv("./iPro_EL/iPro_EL_hs_TA_test.csv")
predicted_hs_TA_iPro_EL = hs_TA_iPro_EL['Proba_R']
printPerformance(TA_label, predicted_hs_TA_iPro_EL, printout=True, csv_export=True, tag='hs_TA', group='iPro_EL', decimal=2)

# HS nonTA
hs_nonTA_iPro_EL= pd.read_csv("./iPro_EL/iPro_EL_hs_nonTA_test.csv")
predicted_hs_nonTA_iPro_EL = hs_nonTA_iPro_EL['Proba_R']
printPerformance(nonTA_label, predicted_hs_nonTA_iPro_EL, printout=True, csv_export=True, tag='hs_nonTA', group='iPro_EL', decimal=2)