# import libraries
import random
import numpy as np
import pandas as pd
import os

#====================================================================================#
# Refine double-spaced file
def refine_data(species='hs', group='TA', seqtype='promoter', export=False, printout=False):
    #-------------------------
    _species = species
    _group   = group
    _seqtype = seqtype
    #-------------------------
    # Refine double-spaced file
    lines = open("./data/{}/{}_{}_{}".format(_species, _species, _seqtype, _group)).readlines()
    new_lines = []
    for line in lines:
        if line != '\n':
            new_lines.append(line)
    #-------------------------
    if export:
        directory = './refined_data/{}/{}'.format(_species, _group)
        if os.path.isdir(directory) == False:
            os.makedirs(directory)
        #-------------------------
        with open('./refined_data/{}/{}/{}_{}_{}.fa'.format(_species, _group, _species, _seqtype, _group), 'w') as f:
            for nline in new_lines:
                f.writelines(nline)
            f.close()
    #-------------------------
    if printout:
        print('Species: {} - Group: {} - Sequence type: {} - Number of sequences: {}'.format(_species, _group, _seqtype, int(len(new_lines)/6)))    
    #-------------------------
    return new_lines

#====================================================================================#
# Extract sequence and store in list
def extract_seq(refined_data, printout=False):
    #-------------------------
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
    joined_seq = list(set(joined_seq))    
    #-------------------------           
    if printout:
        print('Number of extracted sequences: {}'.format(len(joined_seq))) 
    #-------------------------
    return joined_seq

#====================================================================================#
# Remove duplicates between 2 sets of sequences
def remove_duplicate(seqlist1, seqlist2, printout):
    _seqlist1 = seqlist1
    _seqlist2 = seqlist2
    for seq in _seqlist1:
        if seq in _seqlist2:
            _seqlist1.remove(seq)
            _seqlist2.remove(seq)
    if printout:
        print('Seqlist1 - Before count: {} - After count: {}'.format(len(seqlist1), len(_seqlist1)))
        print('Seqlist2 - Before count: {} - After count: {}'.format(len(seqlist2), len(_seqlist2)))
    return _seqlist1, _seqlist2

#====================================================================================#
def build_negative(pos_sample_list, species=None, group=None, seed=0, printout=False):
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
    rebuilt_seq_list = list(set(rebuilt_seq_list))
    #-------------------------
    # Export data
    if species  != None and group != None:
        species_ = species
        group_   = group
        negseq_list = []
        #-------------------------
        for i_seq in range(len(rebuilt_seq_list)):
            seq = rebuilt_seq_list[i_seq]
            if i_seq != len(rebuilt_seq_list)-1:
                negseq = ">neg_{}".format(i_seq+1) + "\n" + seq[:60] + "\n" + seq[60:120] + "\n" + seq[120:180] + "\n" + seq[180:240] + "\n" + seq[240:] + "\n"   
            else:
                negseq = ">neg_{}".format(i_seq+1) + "\n" + seq[:60] + "\n" + seq[60:120] + "\n" + seq[120:180] + "\n" + seq[180:240] + "\n" + seq[240:]
            negseq_list.append(negseq)
        #-------------------------
        directory = './refined_data/{}/{}'.format(species, group)
        if os.path.isdir(directory) == False:
            os.makedirs(directory)
        #-------------------------
        with open('./refined_data/{}/{}/{}_fakenonpromoter_{}.fa'.format(species_, group_, species_, group_), 'w') as f:
            for negseq in negseq_list:
                f.writelines(negseq)
            f.close()
    #-------------------------
    if printout:
        print('Number of fake sequence: {}'.format(len(rebuilt_seq_list)))
    return rebuilt_seq_list

#====================================================================================#
def split_data(promoter_list, nonpromoter_list, fakepromoter_list, test_num, val_num, seed=0, printout=False):
    #-------------------------
    _promoter_list     = promoter_list
    _nonpromoter_list  = nonpromoter_list
    _fakepromoter_list = fakepromoter_list
    #-------------------------
    np.random.seed(seed)
    random.shuffle(_promoter_list)
    random.shuffle(_nonpromoter_list)
    random.shuffle(_fakepromoter_list)
    #-------------------------
    # Get test sets
    promoter_indices_test    = np.random.choice(np.arange(0, len(_promoter_list)),    size=test_num, replace=False)
    nonpromoter_indices_test = np.random.choice(np.arange(0, len(_nonpromoter_list)), size=test_num, replace=False)
    test_list_pos, test_list_neg = [], []
    for idx in promoter_indices_test:
        test_list_pos.append(_promoter_list[idx])
    for idx in nonpromoter_indices_test:
        test_list_neg.append(_nonpromoter_list[idx])
    test_list = test_list_pos + test_list_neg
    #-------------------------
    # Get validation sets
    _promoter_list_v    = list(set(_promoter_list)    - set(test_list_pos))
    _nonpromoter_list_v = list(set(_nonpromoter_list) - set(test_list_neg))
    promoter_indices_val    = np.random.choice(np.arange(0, len(_promoter_list_v)),    size=val_num, replace=False)
    nonpromoter_indices_val = np.random.choice(np.arange(0, len(_nonpromoter_list_v)), size=val_num, replace=False)
    val_list_pos, val_list_neg = [], []
    for idx in promoter_indices_val:
        val_list_pos.append(_promoter_list_v[idx])
    for idx in nonpromoter_indices_val:
        val_list_neg.append(_nonpromoter_list_v[idx])
    val_list = val_list_pos + val_list_neg
    #-------------------------
    # Get training sets
    _promoter_list_t = list(set(_promoter_list_v) - set(val_list_pos))
    train_list_neg, train_list = [], []
    fakepromoter_indices_train = np.random.choice(np.arange(0, len(_fakepromoter_list)), size=len(_promoter_list_t), replace=False)
    for idx in fakepromoter_indices_train:
        train_list_neg.append(_fakepromoter_list[idx])
    train_list_pos = _promoter_list_t
    train_list = train_list_pos + train_list_neg
    #-------------------------
    if printout:
        print('Number of training data: {} \n Number of validation data: {} \n Number of test data: {}'.format(len(train_list), len(val_list), len(test_list)))
    #-------------------------
    return train_list, val_list, test_list

#====================================================================================#
class convert_seq:
    def __init__(self, seqlist, species, group, sample_class):
        self._seqlist      = seqlist
        self._species      = species
        self._group        = group
        self._sample_class = sample_class
        
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
    def seq2index(self): 
        joined_seq = self._seqlist
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
        if self._species != None and self._group != None and self._sample_class != None:
            #-------------------------
            directory = './indexed_data/{}/{}'.format(self._species, self._group)
            if os.path.isdir(directory) == False:
                os.makedirs(directory)
            #-------------------------
            np.save("./indexed_data/{}/{}/{}_{}_{}.npy".format(self._species, self._group, self._sample_class, self._species, self._group), index_seq_array)
        #-------------------------
        return index_seq_array
    
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#    
    def seq2fasta(self): 
        #-------------------------
        directory = './fasta_data/{}/{}'.format(self._species, self._group)
        if os.path.isdir(directory) == False:
            os.makedirs(directory)
        #-------------------------
        if self._species != None and self._group != None and self._sample_class != None:
            with open("./fasta_data/{}/{}/{}_{}_{}.fasta".format(self._species, self._group, self._sample_class, self._species, self._group), 'w') as f:
                for n in range(len(self._seqlist)):
                    if n == len(self._seqlist) - 1:
                        line = '>seq_' + str(n) + '_' + self._sample_class + '\n' + self._seqlist[n]
                    else:
                        line = '>seq_' + str(n) + '_' + self._sample_class + '\n' + self._seqlist[n] + '\n'
                    f.writelines(line)
                f.close()

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
    def seq2csv(self): 
        #-------------------------
        directory = './csv_data/{}/{}'.format(self._species, self._group)
        if os.path.isdir(directory) == False:
            os.makedirs(directory)
        #-------------------------
        if self._species != None and self._group != None and self._sample_class != None:
            df = pd.DataFrame(zip(self._seqlist, [1]*int(len(self._seqlist)/2)+[0]*int(len(self._seqlist)/2)), columns=['sequence', 'class'])
            df.to_csv("./csv_data/{}/{}/{}_{}_data_{}.csv".format(self._species, self._group, self._species, self._group, self._sample_class), index=False) 
                      