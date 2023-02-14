# iPromoter-Seqvec: Identifying Promoters using Bi-LSTM and Sequence-embedded Features

#### T-H Nguyen-Vo, Q. H. Trinh, L. Nguyen, P-U. Nguyen-Hoang, [S. Rahardja](http://www.susantorahardja.com/)*, [B. P. Nguyen](https://homepages.ecs.vuw.ac.nz/~nguyenb5/about.html)∗

![alt text](https://github.com/mldlproject/2022-iPromoter-Seqvec/blob/main/iPromoter_Seqvec_abs0.svg)

## Motivation
Promoters, non-coding DNA sequences located at upstream regions of the transcription start site of genes/gene clusters, are essential
regulatory elements for the initiation and regulation of transcriptional processes. Furthermore, identifying promoters in DNA sequences and genomes significantly
contributes to discovering entire structures of genes of interest. Therefore, exploration of promoter regions is one of the most imperative topics in molecular
genetics and biology. Besides experimental techniques, computational methods have been developed to predict promoters. In this study, we propose
iPromoter-Seqvec – an efficient computational model to predict TATA and non-TATA promoters in humans and mice using the bidirectional long short-term
memory neural networks in combination with sequence-embedded features extracted from promoter sequences. The promoter and non-promoter sequences
were retrieved from the Eukaryotic Promoter database and then were refined to create four benchmark datasets for humans and mice.

## Results
The area under the receiver operating characteristic curve (AUCROC) and the area under the precision-recall curve (AUCPR) were used as two key
metrics to evaluate model performance. Results on independent test sets showed that iPromoter-Seqvec outperformed other state-of-the-art methods with
AUCROC values ranging from 0.85 to 0.99 and AUCPR values ranging from 0.86 to 0.99. Models predicting TATA promoters in both species had slightly higher
predictive power compared to those predicting non-TATA promoters. With a novel idea of constructing artificial non-promoter sequences based on promoter
sequences, our models were able to learn highly specific characteristics discriminating promoters from non-promoters to improve predictive efficiency.


## Availability and implementation
Source code and data are available on [GitHub](https://github.com/mldlproject/2022-iPromoter-Seqvec)

## Web-based Application
- Source 1: [Click here](http://124.197.54.240:5001/)
- Source 2: [Click here](http://14.177.208.167:5001/)

## Contact 
[Go to contact information](https://homepages.ecs.vuw.ac.nz/~nguyenb5/contact.html)
