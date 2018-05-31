# Speaker dependent Speech Enhancement Using DNN
This project is built on [sednn](https://github.com/yongxuUSTC/sednn/issues) by YONG XU & QIUQIANG KONG. For more information and how to use [click me](README_old.md)

This model is trained by massive data by single speaker, mixed with background noise and interfering speech.

[1] Q. Wang, J. Du, L. R. Dai and C. H. Lee, "A Multiobjective Learning and Ensembling Approach to High-Performance Speech Enhancement With Compact Neural Network Architectures," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 26, no. 7, pp. 1185-1197, July 2018.

## Data
The model is trained using [audio books](https://librivox.org) , each chapter of audio book is roughly [split](http://librosa.github.io/librosa/generated/librosa.effects.split.html#librosa.effects.split) into sentences, and 94 of 100 noise by [Hu](http://web.cse.ohio-state.edu/pnl/corpus/HuNonspeech/HuCorpus.html). Each sentence in training set is mixed with one sentence of another reader's book, and one randomly picked noise from training noise, repeating or truncating as necessary.

## Evaluation
Original project uses [PESQ](https://www.itu.int/rec/T-REC-P.862-200102-I/en) score as evaluation metric. But it seems not working if there is interfering speech. I am looking for a subjective evaluation metric working for both background noise and interfering speech. Maybe python library [mir_eval]((https://www.itu.int/rec/T-REC-P.862-200102-I/en)) is a candidate.

## Requirement
pip install -r requirement.txt

## Pipeline
1. create mixture csv: X utterance is mixed with Y noise and Z utterance of interfering from onset to offset .
2. Extract mixture, clean, noise+interfere spectrogram to pickle.
3. Calculate feature, target (spectrogram, MFCC, GFCC, IRM) from pickle in previous step. Then dump all feature to disk.
4. load all feature from disk to memory, start training.
5. Inference on test set.
6. Evaluate performance on test set with PESQ tool.

## Limitation
Due to massive amount of data from one person, load all feature to memory is not practicable. Also collect massive data from one user is not practicable.
### Data workaround
Ideal of clustering and bracket.
Train 10 models with 10 different speakers with massive data for each one. When a new user use this app, use something like nearest neighbor to find one speaker in training set that the user's voice most closed to. Then we can use this speaker's model to approximate this user, and enhance the speech.
#### Limitation
Impossible to find a few speakers that are representative, hard to find distance between speakers' voices in high dimension. We need to learn through data to find bases for these voice, which lead to speaker adaptation in SAT branch
### Memory
Instead of loading all samples to memory, load a small batch of samples each time. This idea is implemented in another branch, but running too slow. Next step would be build a pipeline for loading data from disk ,gradient descend and evaluation. 