# import libraries
from utils import *

#====================================================================================#
# 1. Refine and extract sequences
# 1.1 Promoter sequences 
mm_promoter_TA    = extract_seq(refine_data(species='mm', group='TA',    seqtype='promoter', export=True, printout=True), printout=True)
mm_promoter_nonTA = extract_seq(refine_data(species='mm', group='nonTA', seqtype='promoter', export=True, printout=True), printout=True)
hs_promoter_TA    = extract_seq(refine_data(species='hs', group='TA',    seqtype='promoter', export=True, printout=True), printout=True)
hs_promoter_nonTA = extract_seq(refine_data(species='hs', group='nonTA', seqtype='promoter', export=True, printout=True), printout=True)

# 1.2 Non-promoter sequences 
mm_nonpromoter_TA    = extract_seq(refine_data(species='mm', group='TA',    seqtype='nonpromoter', export=True, printout=True), printout=True)
mm_nonpromoter_nonTA = extract_seq(refine_data(species='mm', group='nonTA', seqtype='nonpromoter', export=True, printout=True), printout=True)
hs_nonpromoter_TA    = extract_seq(refine_data(species='hs', group='TA',    seqtype='nonpromoter', export=True, printout=True), printout=True)
hs_nonpromoter_nonTA = extract_seq(refine_data(species='hs', group='nonTA', seqtype='nonpromoter', export=True, printout=True), printout=True)

# 1.3 Remove dupplicates between promoters and non-promoters
mm_promoter_TA,     mm_nonpromoter_TA = remove_duplicate(mm_promoter_TA,    mm_nonpromoter_TA,    printout=True)
mm_promoter_nonTA , mm_promoter_nonTA = remove_duplicate(mm_promoter_nonTA, mm_nonpromoter_nonTA, printout=True)
hs_promoter_TA ,    hs_nonpromoter_TA = remove_duplicate(hs_promoter_TA,    hs_nonpromoter_TA,    printout=True)
hs_promoter_nonTA , hs_promoter_nonTA = remove_duplicate(hs_promoter_nonTA, hs_nonpromoter_nonTA, printout=True)

#====================================================================================#
# 2. Construct artificial non-promoter sequences
mm_fake_nonpromoter_TA    = build_negative(mm_promoter_TA,    species='mm', group='TA',    printout=True)
mm_fake_nonpromoter_nonTA = build_negative(mm_promoter_nonTA, species='mm', group='nonTA', printout=True)
hs_fake_nonpromoter_TA    = build_negative(hs_promoter_TA,    species='hs', group='TA',    printout=True)
hs_fake_nonpromoter_nonTA = build_negative(hs_promoter_nonTA, species='hs', group='nonTA', printout=True)

#====================================================================================#
# 3. Create train - val - test sets
mm_TA_data    = split_data(mm_promoter_TA,    mm_nonpromoter_TA,    mm_fake_nonpromoter_TA,    test_num=250,  val_num=200,  printout=True)      
mm_nonTA_data = split_data(mm_promoter_nonTA, mm_nonpromoter_nonTA, mm_fake_nonpromoter_nonTA, test_num=2500, val_num=2000, printout=True)   
hs_TA_data    = split_data(hs_promoter_TA,    hs_nonpromoter_TA,    hs_fake_nonpromoter_TA,    test_num=250,  val_num=200,  printout=True)      
hs_nonTA_data = split_data(hs_promoter_nonTA, hs_nonpromoter_nonTA, hs_fake_nonpromoter_nonTA, test_num=2500, val_num=2000, printout=True)   
  
#====================================================================================#
sample_class   = ['train', 'val', 'test']
promoter_class = ['TA', 'nonTA']
species_class  = ['mm', 'hs']
for s in species_class:
    for p in promoter_class:
        if s == 'mm' and p == 'TA':
            seq_data = mm_TA_data
        elif s == 'mm' and p == 'nonTA':
            seq_data = mm_nonTA_data
        elif s == 'hs' and p == 'TA':
            seq_data = hs_TA_data
        else:
            seq_data = hs_nonTA_data    
        for i in range(3):
            convert_seq(seq_data[i], species=s, group=p, sample_class=sample_class[i]).seq2index()   
            convert_seq(seq_data[i], species=s, group=p, sample_class=sample_class[i]).seq2fasta()  
            convert_seq(seq_data[i], species=s, group=p, sample_class=sample_class[i]).seq2csv() 
