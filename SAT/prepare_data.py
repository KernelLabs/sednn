"""
Summary:  Prepare data. 
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: - 
"""
import os
import sys
import soundfile
import numpy as np
import argparse
import csv
import time
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import cPickle
import h5py
from sklearn import preprocessing
import librosa
import config as cfg
import feature_extractor


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs


def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)


###
def create_mixture_csv(args):
    """Create csv containing mixture information. 
    Each line in the .csv file contains [speech_name, noise_name, noise_onset, noise_offset]
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of speech data. 
      noise_dir: str, path of noise data. 
      data_type: str, 'train' | 'test'. 
      magnification: int, only used when data_type='train', number of noise 
          selected to mix with a speech. E.g., when magnication=3, then 4620
          speech with create 4620*3 mixtures. magnification should not larger 
          than the species of noises. 
    """
    workspace = args.workspace
    speech_dir = args.speech_dir
    noise_dir = args.noise_dir
    data_type = args.data_type
    magnification = args.magnification
    fs = cfg.sample_rate

    speech_names = [na for na in os.listdir(speech_dir) if na.lower().endswith(".wav")]
    noise_names = [na for na in os.listdir(noise_dir) if na.lower().endswith(".wav")]
    interfere_names = speech_names[:]

    rs = np.random.RandomState(0)
    out_csv_path = os.path.join(workspace, "mixture_csvs", "%s.csv" % data_type)
    create_folder(os.path.dirname(out_csv_path))

    cnt = 0
    f = open(out_csv_path, 'w')
    f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
    "speech_name", "noise_name", "interfere_name", "noise_onset", "noise_offset", "interfere_onset",
    "interfere_offset"))
    for speech_na in speech_names:
        # Read speech. 
        speech_path = os.path.join(speech_dir, speech_na)
        (speech_audio, _) = read_audio(speech_path, fs)
        len_speech = len(speech_audio)

        # For training data, mix each speech with randomly picked #magnification noises. 
        if data_type == 'train':
            selected_noise_names = rs.choice(noise_names, size=magnification, replace=False)
        # For test data, mix each speech with all noises. 
        elif data_type == 'test':
            selected_noise_names = noise_names
        else:
            raise Exception("data_type must be train | test!")
        interfere_na = rs.choice(interfere_names)
        while interfere_na[:9] == speech_na[:9]: # random select interfere speech, different speaker
            interfere_na = rs.choice(interfere_names)

        # Mix one speech with different noises many times. 
        for noise_na in selected_noise_names:
            noise_path = os.path.join(noise_dir, noise_na)
            (noise_audio, _) = read_audio(noise_path, fs)
            len_noise = len(noise_audio)
            if len_noise <= len_speech:
                noise_onset = 0
                noise_offset = len_speech
            # If noise longer than speech then randomly select a segment of noise.
            else:
                noise_onset = rs.randint(0, len_noise - len_speech, size=1)[0]
                noise_offset = noise_onset + len_speech

            interfere_path = os.path.join(speech_dir, interfere_na)
            (interfere_audio, _) = read_audio(interfere_path, fs)
            len_interfere = len(interfere_audio)
            if len_interfere <= len_speech:
                interfere_onset = 0
                interfere_offset = len_speech
            # If interfere longer than speech then randomly select a segment of interfere.
            else:
                interfere_onset = rs.randint(0, len_interfere - len_speech, size=1)[0]
                interfere_offset = interfere_onset + len_speech

            if cnt % 100 == 0:
                print cnt

            cnt += 1
            f.write("%s\t%s\t%s\t%d\t%d\t%d\t%d\n" % (
            speech_na, noise_na, interfere_na, noise_onset, noise_offset, interfere_onset, interfere_offset))
    f.close()
    print(out_csv_path)
    print("Create %s mixture csv finished!" % data_type)


###
def calculate_mixture_features(args):
    """Calculate spectrogram for mixed, speech and noise audio. Then write the 
    features to disk. 
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of speech data. 
      noise_dir: str, path of noise data. 
      data_type: str, 'train' | 'test'. 
      snr: float, signal to noise ratio to be mixed with noise.
      interfere_snr: float, signal to noise ratio to be mixed with interfering.
    """
    workspace = args.workspace
    speech_dir = args.speech_dir
    noise_dir = args.noise_dir
    data_type = args.data_type
    snr = args.snr
    interfere_snr = args.interfere_snr
    fs = cfg.sample_rate

    # Open mixture csv. 
    mixture_csv_path = os.path.join(workspace, "mixture_csvs", "%s.csv" % data_type)
    with open(mixture_csv_path, 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)

    t1 = time.time()
    cnt = 0
    total_frame = 0
    for i1 in xrange(1, len(lis)):
        [speech_na, noise_na, interfere_na, noise_onset, noise_offset, interfere_onset, interfere_offset] = lis[i1]
        noise_onset = int(noise_onset)
        noise_offset = int(noise_offset)
        interfere_onset = int(interfere_onset)
        interfere_offset = int(interfere_offset)

        # Read speech audio. 
        speech_path = os.path.join(speech_dir, speech_na)
        (speech_audio, _) = read_audio(speech_path, target_fs=fs)

        # Read noise audio.
        noise_audio = load_fix_len_audio(noise_dir, noise_na, len(speech_audio), fs, noise_onset, noise_offset)
        interfere_audio = load_fix_len_audio(speech_dir, interfere_na, len(speech_audio), fs, interfere_onset,
                                             interfere_offset)

        # Scale speech to given snr.
        scaler = get_amplitude_scaling_factor(speech_audio, noise_audio, snr=snr)
        speech_audio *= scaler
        # Get normalized mixture, speech, noise. 
        (mixed_audio, speech_audio, noise_audio, alpha) = additive_mixing(speech_audio, noise_audio)
        scaler = get_amplitude_scaling_factor(mixed_audio, interfere_audio, snr=interfere_snr)
        mixed_audio *= scaler
        (mixed_audio, _, interfere_audio, alpha) = additive_mixing(mixed_audio, interfere_audio)

        # Write out mixed audio. 
        out_bare_na = os.path.join("%s.%s" % (os.path.splitext(speech_na)[0], os.path.splitext(noise_na)[0]))
        out_audio_path = os.path.join(workspace, "mixed_audios", data_type, "%ddb" % int(snr), "%s.wav" % out_bare_na)
        create_folder(os.path.dirname(out_audio_path))
        write_audio(out_audio_path, mixed_audio, fs)

        # Extract spectrogram. 
        mixed_complx_x = calc_sp(mixed_audio, mode='complex')
        speech_x = calc_sp(speech_audio, mode='magnitude')
        noise_x = calc_sp(noise_audio + interfere_audio, mode='magnitude')  # Interfere as noise
        total_frame += mixed_complx_x.shape[0]

        # Write out features. 
        out_feat_path = os.path.join(workspace, "features", "spectrogram",
                                     data_type, "%ddb" % int(snr), "%s.p" % out_bare_na)
        create_folder(os.path.dirname(out_feat_path))
        data = [mixed_complx_x, speech_x, noise_x, alpha, out_bare_na, speech_na]
        cPickle.dump(data, open(out_feat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

        # Print. 
        if cnt % 100 == 0:
            print(cnt)
            sys.stdout.flush()

        cnt += 1

    print("Extracting feature time: %s" % (time.time() - t1))
    total_frame_path = os.path.join(workspace, "total_frame", "%s.p" % data_type)
    create_folder(os.path.dirname(total_frame_path))
    cPickle.dump(total_frame, open(total_frame_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)


def load_fix_len_audio(dir, name, length, fs, onset, offset):
    path = os.path.join(dir, name)
    (audio, _) = read_audio(path, target_fs=fs)

    # Repeat noise to the same length as speech.
    if len(audio) < length:
        n_repeat = int(np.ceil(float(length) / float(len(audio))))
        audio_ex = np.tile(audio, n_repeat)
        audio = audio_ex[0: length]
    # Truncate noise to the same length as speech.
    else:
        audio = audio[onset: offset]
    return audio


def rms(y):
    """Root mean square. 
    """
    return np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))


def get_amplitude_scaling_factor(s, n, snr, method='rms'):
    """Given s and n, return the scaler s according to the snr. 
    
    Args:
      s: ndarray, source1. 
      n: ndarray, source2. 
      snr: float, SNR. 
      method: 'rms'. 
      
    Outputs:
      float, scaler. 
    """
    original_sn_rms_ratio = rms(s) / rms(n)
    target_sn_rms_ratio = 10. ** (float(snr) / 20.)  # snr = 20 * lg(rms(s) / rms(n))
    signal_scaling_factor = target_sn_rms_ratio / original_sn_rms_ratio
    return signal_scaling_factor


def additive_mixing(s, n):
    """Mix normalized source1 and source2. 
    
    Args:
      s: ndarray, source1. 
      n: ndarray, source2. 
      
    Returns:
      mix_audio: ndarray, mixed audio. 
      s: ndarray, pad or truncated and scalered source1. 
      n: ndarray, scaled source2. 
      alpha: float, normalize coefficient. 
    """
    mixed_audio = s + n

    alpha = 1. / np.max(np.abs(mixed_audio))
    mixed_audio *= alpha
    s *= alpha
    n *= alpha
    return mixed_audio, s, n, alpha


def calc_sp(audio, mode):
    """Calculate spectrogram. 
    
    Args:
      audio: 1darray. 
      mode: string, 'magnitude' | 'complex'
    
    Returns:
      spectrogram: 2darray, (n_time, n_freq). 
    """
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    ham_win = np.hamming(n_window)
    [f, t, x] = signal.spectral.spectrogram(
        audio,
        window=ham_win,
        nperseg=n_window,
        noverlap=n_overlap,
        detrend=False,
        return_onesided=True,
        mode=mode)
    x = x.T
    if mode == 'magnitude':
        x = x.astype(np.float32)
    elif mode == 'complex':
        x = x.astype(np.complex64)
    else:
        raise Exception("Incorrect mode!")
    return x


###
def pack_features(args):
    """Load all features, apply log and conver to 3D tensor, write out to .h5 file. 
    
    Args:
      workspace: str, path of workspace. 
      data_type: str, 'train' | 'test'. 
      snr: float, signal to noise ratio to be mixed. 
      n_concat: int, number of frames to be concatenated. 
      n_hop: int, hop frames. 
    """
    workspace = args.workspace
    data_type = args.data_type
    snr = args.snr
    n_concat = args.n_concat
    n_hop = args.n_hop
    n_noise_frame = args.noise_frame
    input_dim1 = (257 + 40 + 30) * 2
    input_dim2 = (257 + 40 + 30)
    out_dim1 = (257 + 40 + 30) * 2
    out_dim1_irm = 257 + 40 + 64
    out_dim2 = (257 + 40 + 30)
    out_dim2_irm = (257 + 40 + 64)

    total_frame_path = os.path.join(workspace, "total_frame", "%s.p" % data_type)
    total_frame = cPickle.load(open(total_frame_path, 'rb'))

    # x_all = []  # (n_segs, n_concat, n_freq)
    # y_all = []  # (n_segs, n_freq)
    # pre-allocate memory for features
    x_all = np.zeros((total_frame, n_concat, input_dim1))
    y_all = np.zeros((total_frame, out_dim1 + out_dim1_irm))
    y2_all = np.zeros((total_frame, out_dim2 + out_dim2_irm))
    x2_all = np.zeros((total_frame, input_dim2))
    adaptive_utterances = np.empty((total_frame), dtype='S10')

    cnt = 0
    t1 = time.time()

    # Load all features. 
    feat_dir = os.path.join(workspace, "features", "spectrogram", data_type, "%ddb" % int(snr))
    names = os.listdir(feat_dir)
    mel_basis = librosa.filters.mel(cfg.sample_rate, cfg.n_window, n_mels=40)
    idx = 0
    for na in names:
        # Load feature. 
        feat_path = os.path.join(feat_dir, na)
        data = cPickle.load(open(feat_path, 'rb'))
        [mixed_complx_x, speech_x, noise_x, alpha, na, speech_na] = data
        input1_3d, input2, out1, out2 = get_input_output_layer(mixed_complx_x, speech_x, noise_x, alpha, n_concat,
                                                               n_noise_frame, n_hop, mel_basis)
        cur_frame = idx + input1_3d.shape[0]
        x_all[idx:cur_frame, :, :] = input1_3d
        x2_all[idx:cur_frame, :] = input2
        y_all[idx:cur_frame, :] = out1
        y2_all[idx:cur_frame, :] = out2
        adaptive_utterances[idx:cur_frame] = speech_na[:9]
        idx = cur_frame

        # Print.
        if cnt % 100 == 0:
            print(cnt)
            sys.stdout.flush()

        # if cnt == 3: break
        cnt += 1

    # x_all = np.concatenate(x_all, axis=0)  # (n_segs, n_concat, n_freq)
    # y_all = np.concatenate(y_all, axis=0)  # (n_segs, n_freq)
    # x2_all = np.concatenate(x2_all, axis=0)  # (n_segs, n_concat, n_freq)
    # y2_all = np.concatenate(y2_all, axis=0)  # (n_segs, n_freq)

    # x_all = log_sp(x_all).astype(np.float32)
    # y_all = log_sp(y_all).astype(np.float32)

    # Write out data to .h5 file. 
    out_path = os.path.join(workspace, "packed_features", "spectrogram", data_type, "%ddb" % int(snr), "data.h5")
    create_folder(os.path.dirname(out_path))
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('x1', data=x_all)
        hf.create_dataset('x2', data=x2_all)
        hf.create_dataset('y1', data=y_all)
        hf.create_dataset('y2', data=y2_all)
        hf.create_dataset('adapt_utter', data=adaptive_utterances)

    print("Write out to %s" % out_path)
    print("Pack features finished! %s s" % (time.time() - t1,))
    sys.stdout.flush()


def get_input_output_layer(mixed_complx_x, speech_x, noise_x, alpha, n_concat, n_noise_frame, n_hop, mel_basis):
    n_pad = (n_concat - 1)
    n = mixed_complx_x.shape[0]

    noisy_lps = np.log((np.abs(mixed_complx_x)) ** 2 + 1e-08)
    static_noise_lps = np.average(noisy_lps[:n_noise_frame, :], axis=0)
    clean_lps = np.log((np.abs(speech_x)) ** 2)
    noise_lps = np.log((np.abs(noise_x)) ** 2)
    noisy_mel_spec = np.dot(mel_basis, np.abs(mixed_complx_x.T))
    clean_mel_spec = np.dot(mel_basis, np.abs(speech_x.T))
    noise_mel_spec = np.dot(mel_basis, np.abs(noise_x.T))
    noisy_mfcc = librosa.feature.mfcc(S=np.log(noisy_mel_spec), n_mfcc=40).T
    clean_mfcc = librosa.feature.mfcc(S=np.log(clean_mel_spec), n_mfcc=40).T
    noise_mfcc = librosa.feature.mfcc(S=np.log(noise_mel_spec), n_mfcc=40).T
    static_noise_mfcc = np.average(noisy_mfcc[:n_noise_frame, :], axis=0)
    gtm = feature_extractor.fft_to_cochleagram(cfg.sample_rate, 0, cfg.sample_rate / 2, cfg.n_window, 64)
    noisy_gf = 1. / cfg.n_window * np.matmul(gtm, np.abs(mixed_complx_x.T))
    clean_gf = 1. / cfg.n_window * np.matmul(gtm, np.abs(speech_x.T))
    noise_gf = 1. / cfg.n_window * np.matmul(gtm, np.abs(noise_x.T))
    dct = librosa.filters.dct(30, 64)
    noisy_gfcc = np.dot(dct, np.power(noisy_gf, 1. / 3)).T
    clean_gfcc = np.dot(dct, np.power(clean_gf, 1. / 3)).T
    noise_gfcc = np.dot(dct, np.power(noise_gf, 1. / 3)).T
    static_noise_gfcc = np.average(noisy_gfcc[:n_noise_frame, :], axis=0).T
    irm = np.abs(speech_x) ** 2 / (np.abs(speech_x) ** 2 + np.abs(noise_x) ** 2)
    irm_mel = (clean_mel_spec / (clean_mel_spec + noise_mel_spec)).T
    irm_gf = (clean_gf / (clean_gf + noise_gf)).T
    input1 = np.hstack([noisy_lps,np.tile(static_noise_lps, (n, 1)), noisy_mfcc, np.tile(static_noise_mfcc, (n, 1)), noisy_gfcc, np.tile(static_noise_gfcc, (n, 1))])
    input1 = pad_head_with_border(input1, n_pad)
    input1_3d = mat_2d_to_3d(input1, agg_num=n_concat, hop=n_hop)

    out1 = np.hstack([clean_lps, noise_lps, clean_mfcc, noise_mfcc, clean_gfcc, noise_gfcc, irm, irm_mel, irm_gf])
    out1 = pad_head_with_border(out1, n_pad)
    out1_3d = mat_2d_to_3d(out1, agg_num=n_concat, hop=n_hop)
    out1 = out1_3d[:, (n_concat - 1), :]

    input2 = np.hstack([noisy_lps, noisy_mfcc, noisy_gfcc])
    input2 = pad_head_with_border(input2, n_pad)
    input2_3d = mat_2d_to_3d(input2, agg_num=n_concat, hop=n_hop)
    input2 = input2_3d[:, (n_concat - 1) / 2, :]

    out2 = np.hstack([clean_lps, clean_mfcc, clean_gfcc, irm, irm_mel, irm_gf])
    out2 = pad_head_with_border(out2, n_pad)
    out2_3d = mat_2d_to_3d(out2, agg_num=n_concat, hop=n_hop)
    out2 = out2_3d[:, (n_concat - 1), :]

    return input1_3d, input2, out1, out2


def log_sp(x):
    return np.log(x + 1e-08)


def mat_2d_to_3d(x, agg_num, hop):
    """Segment 2D array to 3D segments. 
    """
    # Pad to at least one block. 
    len_x, n_in = x.shape
    if (len_x < agg_num):
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))

    # Segment 2d to 3d. 
    len_x = len(x)
    i1 = 0
    x3d = []
    while (i1 + agg_num <= len_x):
        x3d.append(x[i1: i1 + agg_num])
        i1 += hop
    return np.array(x3d)


def pad_with_border(x, n_pad):
    """Pad the begin and finish of spectrogram with border frame value. 
    """
    x_pad_list = [x[0:1]] * n_pad + [x] + [x[-1:]] * n_pad
    return np.concatenate(x_pad_list, axis=0)


def pad_head_with_border(x, n_pad):
    """Pad the begin and finish of spectrogram with border frame value.
    """
    x_pad_list = [x[0:1]] * n_pad + [x]
    return np.concatenate(x_pad_list, axis=0)


###
def compute_scaler(args):
    """Compute and write out scaler of data. 
    """
    workspace = args.workspace
    data_type = args.data_type
    snr = args.snr

    # Load data. 
    t1 = time.time()
    hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", data_type, "%ddb" % int(snr), "data.h5")
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x1')
        x = np.array(x)  # (n_segs, n_concat, n_freq)

    # Compute scaler. 
    (n_segs, n_concat, n_freq) = x.shape
    x2d = x.reshape((n_segs * n_concat, n_freq))
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(x2d)
    print(scaler.mean_)
    print(scaler.scale_)

    # Write out scaler. 
    out_path = os.path.join(workspace, "packed_features", "spectrogram", data_type, "%ddb" % int(snr), "scaler.p")
    create_folder(os.path.dirname(out_path))
    pickle.dump(scaler, open(out_path, 'wb'))

    print("Save scaler to %s" % out_path)
    print("Compute scaler finished! %s s" % (time.time() - t1,))


def scale_on_2d(x2d, scaler):
    """Scale 2D array data. 
    """
    return scaler.transform(x2d)


def scale_on_3d(x3d, scaler):
    """Scale 3D array data. 
    """
    (n_segs, n_concat, n_freq) = x3d.shape
    x2d = x3d.reshape((n_segs * n_concat, n_freq))
    x2d = scaler.transform(x2d)
    x3d = x2d.reshape((n_segs, n_concat, n_freq))
    return x3d


def inverse_scale_on_2d(x2d, scaler):
    """Inverse scale 2D array data. 
    """
    return x2d * scaler.scale_[None, :] + scaler.mean_[None, :]


###
def load_hdf5(hdf5_path):
    """Load hdf5 data. 
    """
    with h5py.File(hdf5_path, 'r') as hf:
        x1 = hf.get('x1')
        y1 = hf.get('y1')
        x2 = hf.get('x2')
        y2 = hf.get('y2')
        adapt_utter = hf.get('adapt_utter')
        x1 = np.array(x1)
        y1 = np.array(y1)
        x2 = np.array(x2)
        y2 = np.array(y2)
        adapt_utter = np.array(adapt_utter)
    return x1, x2, y1, y2, adapt_utter


def np_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def calculate_adaptive_utterance_features(args):
    """Compute and write out adaptive utterance spectrogram
    """
    workspace = args.workspace
    data_type = args.data_type
    adaptive_utterance_dir = args.ada_utt_dir
    names = os.listdir(adaptive_utterance_dir)
    names = set([name[:9] for name in names])
    all_features = dict()
    max_len = 0
    for name in names:
        path = os.path.join(adaptive_utterance_dir, name)
        (audio1, _) = read_audio(path + '_SA1.WAV', cfg.sample_rate)
        audio_spec1 = calc_sp(audio1, mode='magnitude')
        (audio2, _) = read_audio(path + '_SA2.WAV', cfg.sample_rate)
        audio_spec2 = calc_sp(audio2, mode='magnitude')
        all_features[name] = np.vstack([audio_spec1, audio_spec2])
        max_len = max(max_len, len(all_features[name]))

    out_feat_path = os.path.join(workspace, "adaptive_utterance", data_type, "adaptive_utterance_spec.p")
    create_folder(os.path.dirname(out_feat_path))
    cPickle.dump(all_features, open(out_feat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

    out_feat_path = os.path.join(workspace, "adaptive_utterance", data_type, "adaptive_utterance_max_len.p")
    create_folder(os.path.dirname(out_feat_path))
    cPickle.dump(max_len, open(out_feat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)


###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_create_mixture_csv = subparsers.add_parser('create_mixture_csv')
    parser_create_mixture_csv.add_argument('--workspace', type=str, required=True)
    parser_create_mixture_csv.add_argument('--speech_dir', type=str, required=True)
    parser_create_mixture_csv.add_argument('--noise_dir', type=str, required=True)
    parser_create_mixture_csv.add_argument('--data_type', type=str, required=True)
    parser_create_mixture_csv.add_argument('--magnification', type=int, default=1)

    parser_calculate_mixture_features = subparsers.add_parser('calculate_mixture_features')
    parser_calculate_mixture_features.add_argument('--workspace', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--noise_dir', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--data_type', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--snr', type=float, required=True)
    parser_calculate_mixture_features.add_argument('--interfere_snr', type=float, required=True)

    parser_calculate_mixture_features = subparsers.add_parser('calculate_adaptive_utterance_features')
    parser_calculate_mixture_features.add_argument('--workspace', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--ada_utt_dir', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--data_type', type=str, required=True)

    parser_pack_features = subparsers.add_parser('pack_features')
    parser_pack_features.add_argument('--workspace', type=str, required=True)
    parser_pack_features.add_argument('--data_type', type=str, required=True)
    parser_pack_features.add_argument('--snr', type=float, required=True)
    parser_pack_features.add_argument('--n_concat', type=int, required=True)
    parser_pack_features.add_argument('--n_hop', type=int, required=True)
    parser_pack_features.add_argument('--noise_frame', type=int, required=True)

    parser_compute_scaler = subparsers.add_parser('compute_scaler')
    parser_compute_scaler.add_argument('--workspace', type=str, required=True)
    parser_compute_scaler.add_argument('--data_type', type=str, required=True)
    parser_compute_scaler.add_argument('--snr', type=float, required=True)

    args = parser.parse_args()
    if args.mode == 'create_mixture_csv':
        create_mixture_csv(args)
    elif args.mode == 'calculate_mixture_features':
        calculate_mixture_features(args)
    elif args.mode == 'calculate_adaptive_utterance_features':
        calculate_adaptive_utterance_features(args)
    elif args.mode == 'pack_features':
        pack_features(args)
    elif args.mode == 'compute_scaler':
        compute_scaler(args)
    else:
        raise Exception("Error!")
