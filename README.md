# iPromoter-Seqvec: Identifying Promoters using Bidirectional LSTM and Sequence-embedded Features


#### T-H Nguyen-Vo, Q. H. Trinh, L. Nguyen, P-U. Nguyen-Hoang, [S. Rahardja](http://www.susantorahardja.com/)*, [B. P. Nguyen](https://homepages.ecs.vuw.ac.nz/~nguyenb5/about.html)∗

![alt text](https://github.com/mldlproject/2022-iPromoter-Seqvec/blob/main/iPromoter_Seqvec_abs0.svg)

## Motivation
Promoters, non-coding DNA sequences located at upstream regions of the transcription start site of genes/gene clusters, are essential regulatory elements for the 
initiation and regulation of transcriptional processes. Furthermore, identifying promoters in the DNA sequences and genomes significantly contributes to discovering 
entire structures of genes of interest. Therefore, the exploration of promoter regions is one of the most imperative topics in molecular genetics and biology. Besides 
experimental techniques, computational methods have been developed to predict promoters. In this study, we propose iPromoter-Seqvec an efficient computational model 
to predict TATA and non-TATA promoters in humans and mice using the bidirectional Long Short-term Memory neural networks in combination with sequence-embedded features 
extracted from promoter sequences. The promoter and non-promoter sequences were retrieved from the Eukaryotic Promoter Database and then refined to create four benchmark 
datasets for humans and mice. 

## Results
Results on independent test sets show that iPromoter-Seqvec outperforms other state-of-the-art methods with AUC-ROC values ranging from 0.85 to 0.99 and AUC-PR values 
ranging from 0.86 to 0.99. Models predicting TATA promoters in both species have slightly higher predictive power compared to those predicting non-TATA promoters. With 
a novel idea of constructing artificial non-promoter sequences based on promoter sequences used for model training, models were forced to learn highly specific 
characteristics discriminating promoters from non-promoters to improve predictive efficiency. iPromoter-Seqvec combining the bidirectional long short term memory 
neural networks and sequence-embedded features is a stable and robust prediction model for predicting promoters. Besides, compared to other state-of-the-art methods, 
our method shows better performance in most evaluation metrics.


## Availability and implementation
Source code and data are available on [GitHub](https://github.com/mldlproject/2022-iPromoter-Seqvec)

## Web-based Application
- Source 1: [Click here](http://14.231.244.182:5001/)
- Source 2: [Click here](http://103.130.219.193:8001/)

## Contact 
[Go to contact information](https://homepages.ecs.vuw.ac.nz/~nguyenb5/contact.html)
