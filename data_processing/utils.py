# import libraries
from os import supports_effective_ids
import numpy as np
import random

#====================================================================================#
# Refine double-spaced file
def refine_data(species='hs', group='TA'):
    # Refine double-spaced file
    lines = open("./data/{}/{}_promoter_{}".format(species, species, group)).readlines()
    new_lines = []
    for line in lines:
        if line != '\n':
            new_lines.append(line)
    #-------------------------    
    with open('./refined_data/{}/{}/{}_promoter_{}.fa'.format(species, group, species, group), 'w') as f:
        for nline in new_lines:
            f.writelines(nline)
        f.close()
    #-------------------------    
    return new_lines

#====================================================================================#
# Extract sequence and store in list
def extract_seq(refined_data):
    lines = refined_data
    #-------------------------
    # Remove sequence's header 
    noheader_seq = []
    n_count = 0
    for nline in lines:
        n_count = n_count + 1
        if nline[0] != ">":
            if n_count  < len(lines):
                noheader_seq.append(nline[:-1])
            else:
                noheader_seq.append(nline) # final line does not have "\n"
    #-------------------------
    # join subsequence and remove space
    k_range = np.arange(0,len(noheader_seq)+5, 5)
    k_count = 0
    joined_seq = []
    for k in range(len(k_range)):
        k_count = k_count + 1
        if k_count < len(k_range):
            seq = "".join(noheader_seq[k_range[k]:k_range[k+1]])
            joined_seq.append(seq)     
    #-------------------------            
    return joined_seq

#====================================================================================#
def build_negative(pos_sample_list, species=None, group=None, seed=0):
    #-------------------------
    pos_seqs = pos_sample_list
    #-------------------------
    # Split sequences in 20 equal fragments
    s_range = np.arange(0,305,15)
    seq_split_total = []
    for seq in pos_seqs:
        s_count = 0
        seq_split = []
        for s in range(len(s_range)):
            s_count = s_count + 1
            if s_count < len(s_range):
                seq_split.append(seq[s_range[s]:s_range[s+1]])
        seq_split_total.append(seq_split)
    #-------------------------
    # Re-index fragments of sequences to create recombinant sequences (negative samples)    
    reindexed_list_total = []
    seed_ = seed
    for i_seq in range(len(seq_split_total)):
        random.seed(i_seq+seed_)
        ilist = list(np.arange(0,20))
        shuffle_list = random.sample(ilist,12)
        conservative_list = list(set(ilist) - set (shuffle_list))
        reindexed_list = ilist
        i_count = 0
        for i in range(len(ilist)):
            if i_count < len(shuffle_list):
                if ilist[i] not in conservative_list:
                    reindexed_list[i] = shuffle_list[i_count]
                    i_count = i_count + 1
        reindexed_list_total.append(reindexed_list)   
    #-------------------------
    # Re-built recombinant sequences (negative samples)    
    rebuilt_seq_list = []    
    for i_index in range(len(reindexed_list_total)):
        initial_seq = seq_split_total[i_index] 
        rebuilt_seq = []
        for j_index in reindexed_list_total[i_index]:
            rebuilt_seq.append(initial_seq[j_index])
        rebuilt_seq_list.append("".join(rebuilt_seq))
    #-------------------------
    # Export data
    if species  != None and group != None:
        species_ = species
        group_   = group
        negseq_list = []
        #-------------------------
        for i_seq in range(len(rebuilt_seq_list)):
            seq = rebuilt_seq_list[i_seq]
            negseq = ">neg_{}".format(i_seq+1) + "\n" + seq[:60] + "\n" + seq[60:120] + "\n" + seq[120:180] + "\n" + seq[180:240] + "\n" + seq[240:] + "\n"   
            negseq_list.append(negseq)
        #-------------------------
        with open('./refined_data/{}/{}/{}_nonpromoter_{}.fa'.format(species_, group_, species_, group_), 'w') as f:
            for negseq in negseq_list:
                f.writelines(negseq)
            f.close()
    #-------------------------
    return rebuilt_seq_list

#====================================================================================#
def seq2index(seq, species=None, group=None, sample_class=None): 
    joined_seq = seq
    #-------------------------
    # Create dict for embedding
    char_list = ['A', 'T', 'G', 'C']
    char_dict = ['X']
    for char1 in char_list:
        for char2 in char_list:
            for char3 in char_list:
                char_dict.append(char1 + char2 + char3)
    #-------------------------
    trip_list_total = []
    for seq in joined_seq:
        trip_list = []
        for p in range(298):
            trip_list.append(seq[0+p:3+p])
        trip_list_total.append(trip_list)
    #-------------------------
    index_seq_list = []
    for tri_list in trip_list_total:
        index_seq = []
        for tri in tri_list:
            if tri[0] != "N" and tri[1] != "N" and tri[2] != "N":  
                index_seq.append(char_dict.index(tri))
            else:
                index_seq.append(0)
        index_seq_list.append(index_seq)
    #-------------------------    
    index_seq_array = np.array(index_seq_list)
    if species != None and group != None and sample_class != None:
        np.save("./indexed_data/{}/{}/{}_{}_{}.npy".format(species, group, sample_class, species, group), index_seq_array)
    #-------------------------
    return index_seq_array
