# Speech Enhancement Using DNN
This project is built on [sednn](https://github.com/yongxuUSTC/sednn/issues) by YONG XU & QIUQIANG KONG. For more information and how to use [click me](README_old.md)

This directory is a benchmark for speaker independent removing background noise. Adopting idea from following paper. Including multi-feature representation: magnitude spectrogram, MFCC, GFCC; multi-object learning: spectrogram and ideal ratio mask; static noise aware; multi-stage network.

[1] Q. Wang, J. Du, L. R. Dai and C. H. Lee, "A Multiobjective Learning and Ensembling Approach to High-Performance Speech Enhancement With Compact Neural Network Architectures," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 26, no. 7, pp. 1185-1197, July 2018.

## Data
The model is trained using [TIMIT](http://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3) 4620 training sentences and 94 of 100 noise by [Hu](http://web.cse.ohio-state.edu/pnl/corpus/HuNonspeech/HuCorpus.html). Each sentence in training set one randomly picked noise from training noise, repeating or truncating noise as necessary.

## Evaluation
Original project uses [PESQ](https://www.itu.int/rec/T-REC-P.862-200102-I/en) score as evaluation metric. The result shows about 2.5 PESQ score on average.

## Requirement
pip install -r requirement.txt