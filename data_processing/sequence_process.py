# import libraries
import numpy as np
from utils import *

#====================================================================================#
# 1. Refine and extract positive sequences
mm_pos_TA    = extract_seq(refine_data(species='mm', group='TA'))
mm_pos_nonTA = extract_seq(refine_data(species='mm', group='nonTA'))
hs_pos_TA    = extract_seq(refine_data(species='hs', group='TA'))
hs_pos_nonTA = extract_seq(refine_data(species='hs', group='nonTA'))

#====================================================================================#
# 2. Construct negative sequences
mm_neg_TA    = build_negative(mm_pos_TA, species='mm',group='TA')
mm_neg_nonTA = build_negative(mm_pos_nonTA, species='mm',group='nonTA')
hs_neg_TA    = build_negative(hs_pos_TA, species='hs', group='TA')
hs_neg_nonTA = build_negative(hs_pos_nonTA, species='hs', group='nonTA')
        
#====================================================================================#
# 3. Convert sequence into index vectors
_ = seq2index(mm_pos_TA, species='mm',group='TA', sample_class='pos')
_ = seq2index(mm_pos_nonTA, species='mm',group='nonTA', sample_class='pos')
_ = seq2index(hs_pos_TA, species='hs',group='TA', sample_class='pos')
_ = seq2index(hs_pos_nonTA, species='hs',group='nonTA', sample_class='pos')

_ = seq2index(mm_neg_TA, species='mm',group='TA', sample_class='neg')
_ = seq2index(mm_neg_nonTA, species='mm',group='nonTA', sample_class='neg')
_ = seq2index(hs_neg_TA, species='hs',group='TA', sample_class='neg')
_ = seq2index(hs_neg_nonTA, species='hs',group='nonTA', sample_class='neg')

