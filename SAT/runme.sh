#!/bin/bash
set -e

MINIDATA=1
if [ $MINIDATA -eq 1 ]; then
  WORKSPACE="workspace"
  mkdir -p $WORKSPACE
  TR_SPEECH_DIR="mini_data/train_speech"
  TR_NOISE_DIR="mini_data/train_noise"
  TR_ADAPTUTTER_DIR="mini_data/train_adapt_utt"
  TE_SPEECH_DIR="mini_data/test_speech"
  TE_NOISE_DIR="mini_data/test_noise"
  TE_ADAPTUTTER_DIR="mini_data/test_adapt_utt"
  echo "Using mini data. "
  ITERATION=1000
else
  WORKSPACE="/home/ubuntu/data/workspace"
  mkdir -p $WORKSPACE
  TR_SPEECH_DIR="/home/ubuntu/data/timit/train"
  TR_NOISE_DIR="/home/ubuntu/data/noise/train3"
  TR_ADAPTUTTER_DIR="/home/ubuntu/data/train_adapt_utt"
  TE_SPEECH_DIR="/home/ubuntu/data/timit/test2"
  TE_NOISE_DIR="/home/ubuntu/data/noise/test3"
  TE_ADAPTUTTER_DIR="/home/ubuntu/data/test_adapt_utt"
  echo "Using full data. "
  ITERATION=100000
fi

# Create mixture csv. 
#rm -r $WORKSPACE
python prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --magnification=2
python prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test

# Calculate mixture features.
TR_SNR=10
TE_SNR=10
INTERFERE_SNR=10
python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --snr=$TR_SNR --interfere_snr=$INTERFERE_SNR
python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test --snr=$TE_SNR --interfere_snr=$INTERFERE_SNR

# Spectrogram for adaptive utterance
python prepare_data.py calculate_adaptive_utterance_features --workspace=$WORKSPACE --ada_utt_dir=$TR_ADAPTUTTER_DIR --data_type=train
python prepare_data.py calculate_adaptive_utterance_features --workspace=$WORKSPACE --ada_utt_dir=$TE_ADAPTUTTER_DIR --data_type=test

# Pack features. 
N_CONCAT=3
N_HOP=1
N_NOISE_FRAME=6
python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=train --snr=$TR_SNR --n_concat=$N_CONCAT --n_hop=$N_HOP --noise_frame=$N_NOISE_FRAME
python prepare_data.py pack_features --workspace=$WORKSPACE --data_type=test --snr=$TE_SNR --n_concat=$N_CONCAT --n_hop=$N_HOP --noise_frame=$N_NOISE_FRAME

# Compute scaler. 
# python prepare_data.py compute_scaler --workspace=$WORKSPACE --data_type=train --snr=$TR_SNR

# Train. 
LEARNING_RATE=1e-4
python main_dnn.py train --workspace=$WORKSPACE --tr_snr=$TR_SNR --te_snr=$TE_SNR --lr=$LEARNING_RATE --iter=$ITERATION

# Plot training stat. 
python evaluate.py plot_training_stat --workspace=$WORKSPACE --tr_snr=$TR_SNR --bgn_iter=0 --fin_iter=$((ITERATION+1)) --interval_iter=100

# Inference, enhanced wavs will be created. 
python main_dnn.py inference --workspace=$WORKSPACE --tr_snr=$TR_SNR --te_snr=$TE_SNR --n_concat=$N_CONCAT --iteration=$ITERATION --n_hop=$N_HOP --noise_frame=$N_NOISE_FRAME 

# Calculate PESQ of all enhanced speech. 
python evaluate.py calculate_pesq --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --te_snr=$TE_SNR

# Calculate overall stats. 
python evaluate.py get_stats

date
~/stop-self.sh
