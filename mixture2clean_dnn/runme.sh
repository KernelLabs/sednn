#!/bin/bash
set -e

MINIDATA=0
if [ $MINIDATA -eq 1 ]; then
  WORKSPACE="workspace"
  mkdir $WORKSPACE
  TR_SPEECH_DIR="mini_data/train_speech"
  TR_NOISE_DIR="mini_data/train_noise"
  TE_SPEECH_DIR="mini_data/test_speech"
  TE_NOISE_DIR="mini_data/test_noise"
  echo "Using mini data. "
else
  WORKSPACE="/home/ubuntu/data/workspace"
  mkdir $WORKSPACE
  TR_SPEECH_DIR="/home/ubuntu/data/timit/train"
  TR_NOISE_DIR="/home/ubuntu/data/noise/train3"
  TE_SPEECH_DIR="/home/ubuntu/data/timit/test2"
  TE_NOISE_DIR="/home/ubuntu/data/noise/test3"
  echo "Using full data. "
fi

# Create mixture csv. 
#rm -r $WORKSPACE
python prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --magnification=4
python prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test

# Calculate mixture features.
TR_SNR=10
TE_SNR=10 
python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --snr=$TR_SNR
python prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test --snr=$TE_SNR

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
ITERATION=100000
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
