# Speaker Adaptive Speech Enhancement Using DNN
This project is built on [sednn](https://github.com/yongxuUSTC/sednn/issues) by YONG XU & QIUQIANG KONG. For more information and how to use [click me](README_old.md)

This project has two parts: a stronger speaker indenpendent speeching enhancement model, and a auxiliary speaker encoding extraction network. They are connected by a speaker adaptation layer. The idea of this project comes from:

[1] Q. Wang, J. Du, L. R. Dai and C. H. Lee, "A Multiobjective Learning and Ensembling Approach to High-Performance Speech Enhancement With Compact Neural Network Architectures," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 26, no. 7, pp. 1185-1197, July 2018.

[2] K. Žmolíková, M. Delcroix, K. Kinoshita, T. Higuchi, A. Ogawa and T. Nakatani, "Learning speaker representation for neural network based multichannel speaker extraction," 2017 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), Okinawa, 2017, pp. 8-15.

[3] Žmolíková, Kateřina & Delcroix, Marc & Kinoshita, Keisuke & Higuchi, Takuya & Ogawa, Atsunori & Nakatani, Tomohiro. (2017). Speaker-Aware Neural Network Based Beamformer for Speaker Extraction in Speech Mixtures. 2655-2659. 10.21437/Interspeech.2017-667. 

## Data
The model is trained using [TIMIT](http://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3) 4620 training sentences and 94 of 100 noise by [Hu](http://web.cse.ohio-state.edu/pnl/corpus/HuNonspeech/HuCorpus.html). With all concatenation of SA1 and SA2 used as speaker adaptive utternce. Each sentence in training set other than SA1, SA2 is randomly mixed with another speaker's utterance, and one randomly picked noise from training noise, repeating or truncating as necessary.

## Evaluation
Original project uses [PESQ](https://www.itu.int/rec/T-REC-P.862-200102-I/en) score as evaluation metric. But it seems not working if there is interfering speech. I am looking for a subjective evaluation metric working for both background noise and interfering speech. Maybe python library [mir_eval]((https://www.itu.int/rec/T-REC-P.862-200102-I/en)) is a candidate.

## Requirement
pip install -r requirement.txt

## Next Step
1. train the speaker adaptive model. Due to time limit, it has not been tested yet. Test i-vector vs speaker encoding network. Tune # of layers in aux layer, and # of factorization in adaptation layer.
2. explore feed data/train pipeline. Now data are loaded to memory together, due to the limitation of memory size, we have to constrain the size of dataset. A feed data by small batch has implenemented in interfere branch. But loading data and training seems to be done in sequence, and it is really slow.
3. [Evaluation metric](#evaluation)