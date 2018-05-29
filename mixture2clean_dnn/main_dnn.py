"""
Summary:  Train, inference and evaluate speech enhancement. 
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: -
"""
import numpy as np
import os
import sys
import pickle
import cPickle
import h5py
import argparse
import time
import glob
import matplotlib.pyplot as plt

import prepare_data as pp_data
import config as cfg
from data_generator import DataGenerator
from spectrogram_to_wave import recover_wav
import librosa
import librosa.display as display

from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras.models import load_model
import tensorflow as tf


def eval(model, gen, x1, x2, y1, y2, name, utts):
    """Validation function. 
    
    Args:
      model: keras model. 
      gen: object, data generator. 
      x: 3darray, input, (n_segs, n_concat, n_freq)
      y: 2darray, target, (n_segs, n_freq)
    """
    pred_all, y_all = [], []

    # Inference in mini batch. 
    for (batch_x, batch_y) in gen.generate([x1, x2, name], [y1, y2], utts):
        pred = model.predict(batch_x)
        pred_all.append(np.hstack(pred))
        y_all.append(np.hstack(batch_y))

    # Concatenate mini batch prediction. 
    pred_all = np.concatenate(pred_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    # Compute loss. 
    loss = pp_data.np_mean_absolute_error(y_all, pred_all)
    return loss


def train(args):
    """Train the neural network. Write out model every several iterations. 
    
    Args:
      workspace: str, path of workspace. 
      tr_snr: float, training SNR. 
      te_snr: float, testing SNR. 
      lr: float, learning rate. 
    """
    print(args)
    workspace = args.workspace
    tr_snr = args.tr_snr
    te_snr = args.te_snr
    lr = args.lr
    iteration = args.iter

    # Load data. 
    t1 = time.time()
    tr_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr), "data.h5")
    te_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "test", "%ddb" % int(te_snr), "data.h5")
    tr_adapt_utt_path = os.path.join(workspace, "adaptive_utterance", "train", "adaptive_utterance_spec.p")
    te_adapt_utt_path = os.path.join(workspace, "adaptive_utterance", "test", "adaptive_utterance_spec.p")
    tr_adapt_utt = cPickle.load(open(tr_adapt_utt_path, 'rb'))
    te_adapt_utt = cPickle.load(open(te_adapt_utt_path, 'rb'))
    tr_adapt_utt_len_path = os.path.join(workspace, "adaptive_utterance", "train", "adaptive_utterance_max_len.p")
    te_adapt_utt_len_path = os.path.join(workspace, "adaptive_utterance", "test", "adaptive_utterance_max_len.p")
    tr_adapt_utt_len = cPickle.load(open(tr_adapt_utt_len_path, 'rb'))
    te_adapt_utt_len = cPickle.load(open(te_adapt_utt_len_path, 'rb'))
    max_len = max(tr_adapt_utt_len, te_adapt_utt_len)
    (tr_x1, tr_x2, tr_y1, tr_y2, tr_name) = pp_data.load_hdf5(tr_hdf5_path)
    (te_x1, te_x2, te_y1, te_y2, te_name) = pp_data.load_hdf5(te_hdf5_path)
    print(tr_x1.shape, tr_y1.shape, tr_x2.shape, tr_y2.shape)
    print(te_x1.shape, te_y1.shape, te_x2.shape, te_y2.shape)
    print("Load data time: %s s" % (time.time() - t1,))

    batch_size = 500
    print("%d iterations / epoch" % int(tr_x1.shape[0] / batch_size))

    # Scale data. 
    if not True:
        t1 = time.time()
        scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr),
                                   "scaler.p")
        scaler = pickle.load(open(scaler_path, 'rb'))
        tr_x1 = pp_data.scale_on_3d(tr_x1, scaler)
        tr_y1 = pp_data.scale_on_2d(tr_y1, scaler)
        te_x1 = pp_data.scale_on_3d(te_x1, scaler)
        te_y1 = pp_data.scale_on_2d(te_y1, scaler)
        tr_x2 = pp_data.scale_on_2d(tr_x2, scaler)
        tr_y2 = pp_data.scale_on_2d(tr_y2, scaler)
        te_x2 = pp_data.scale_on_2d(te_x2, scaler)
        te_y2 = pp_data.scale_on_2d(te_y2, scaler)
        print("Scale data time: %s s" % (time.time() - t1,))

    # Debug plot. 
    if False:
        plt.matshow(tr_x[0: 1000, 0, :].T, origin='lower', aspect='auto', cmap='jet')
        plt.show()
        pause

    # Build model
    (_, n_concat, n_freq) = tr_x1.shape
    n_hid = 2048
    input_dim1 = (257 + 40 + 30) * 2
    input_dim2 = (257 + 40 + 30)
    out_dim1 = (257 + 40 + 30) * 2
    out_dim1_irm = 257 + 40 + 64
    out_dim2 = (257 + 40 + 30)
    out_dim2_irm = (257 + 40 + 64)
    num_fact = 30

    def multiplication(pair_tensors):
        x, y = pair_tensors
        return K.sum(tf.multiply(y, K.expand_dims(x, -1)), axis=1)

    adapt_input = Input(shape=(None,), name='adapt_input')
    layer = Reshape((-1, 257), name='reshape')(adapt_input)
    layer = Dense(512, activation='relu', name='adapt_dense1')(layer)
    layer = Dense(512, activation='relu', name='adapt_dense2')(layer)
    layer = Dense(num_fact, activation='softmax', name='adapt_out')(layer)
    alpha = Lambda(lambda x: K.sum(x, axis=1), output_shape=(num_fact,), name='sequence_summing')(layer)
    # alpha2 = Lambda(lambda x:K.repeat_elements(x,n_hid,0),output_shape=(num_fact*n_hid,),name='repeat')(alpha)
    input1 = Input(shape=(n_concat, input_dim1), name='input1')
    layer = Flatten(name='flatten')(input1)
    layer = Dense(n_hid * num_fact, name='dense0')(layer)
    layer = Reshape((num_fact, n_hid), name='reshape2')(layer)
    layer = Lambda(multiplication, name='multiply')([alpha, layer])
    # layer = Multiply()([alpha2,layer])
    # layer = Reshape((num_fact,n_hid))(layer)

    layer = Dense(n_hid, activation='relu', name='dense1')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(n_hid, activation='relu', name='dense2')(layer)
    layer = Dropout(0.2)(layer)
    partial_out1 = Dense(out_dim1, name='1_out_linear')(layer)
    partial_out1_irm = Dense(out_dim1_irm, name='1_out_irm', activation='sigmoid')(layer)
    out1 = concatenate([partial_out1, partial_out1_irm], name='out1')
    input2 = Input(shape=(input_dim2,), name='input2')
    layer = concatenate([input2, out1], name='merge')
    layer = Dense(n_hid, activation='relu', name='dense3')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(n_hid, activation='relu', name='dense4')(layer)
    layer = Dropout(0.2)(layer)
    partial_out2 = Dense(out_dim2, name='2_out_linear')(layer)
    partial_out2_irm = Dense(out_dim2_irm, name='2_out_irm', activation='sigmoid')(layer)
    out2 = concatenate([partial_out2, partial_out2_irm], name='out2')
    model = Model(inputs=[input1, input2, adapt_input], outputs=[out1, out2])

    model.summary()
    sys.stdout.flush()
    model.compile(loss='mean_absolute_error',
                  optimizer=Adam(lr=lr, epsilon=1e-03))
    # Data generator.
    tr_gen = DataGenerator(batch_size=batch_size, type='train', max_len=max_len)
    eval_te_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100, max_len=max_len)
    eval_tr_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100, max_len=max_len)

    # Directories for saving models and training stats
    model_dir = os.path.join(workspace, "models", "%ddb" % int(tr_snr))
    pp_data.create_folder(model_dir)

    stats_dir = os.path.join(workspace, "training_stats", "%ddb" % int(tr_snr))
    pp_data.create_folder(stats_dir)

    # Print loss before training. 
    iter = 0
    tr_loss = eval(model, eval_tr_gen, tr_x1, tr_x2, tr_y1, tr_y2, tr_name, tr_adapt_utt)
    te_loss = eval(model, eval_te_gen, te_x1, te_x2, te_y1, te_y2, te_name, te_adapt_utt)
    print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))

    # Save out training stats. 
    stat_dict = {'iter': iter,
                 'tr_loss': tr_loss,
                 'te_loss': te_loss, }
    stat_path = os.path.join(stats_dir, "%diters.p" % iter)
    cPickle.dump(stat_dict, open(stat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

    # Train. 
    t1 = time.time()
    for (batch_x, batch_y) in tr_gen.generate([tr_x1, tr_x2, tr_name], [tr_y1, tr_y2], tr_adapt_utt):
        loss = model.train_on_batch(batch_x, batch_y)
        iter += 1

        # Validate and save training stats. 
        if iter % 100 == 0:
            tr_loss = eval(model, eval_tr_gen, tr_x1, tr_x2, tr_y1, tr_y2, tr_name, tr_adapt_utt)
            te_loss = eval(model, eval_te_gen, te_x1, te_x2, te_y1, te_y2, te_name, te_adapt_utt)
            print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))
            sys.stdout.flush()

            # Save out training stats. 
            stat_dict = {'iter': iter,
                         'tr_loss': tr_loss,
                         'te_loss': te_loss, }
            stat_path = os.path.join(stats_dir, "%diters.p" % iter)
            cPickle.dump(stat_dict, open(stat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

        # Save model. 
        if iter % (iteration / 20) == 0:
            model_path = os.path.join(model_dir, "md_%diters.h5" % iter)
            model.save(model_path)
            print("Saved model to %s" % model_path)

        if iter == iteration + 1:
            break

    print("Training time: %s s" % (time.time() - t1,))


def inference(args):
    """Inference all test data, write out recovered wavs to disk. 
    
    Args:
      workspace: str, path of workspace. 
      tr_snr: float, training SNR. 
      te_snr: float, testing SNR. 
      n_concat: int, number of frames to concatenta, should equal to n_concat 
          in the training stage. 
      iter: int, iteration of model to load. 
      visualize: bool, plot enhanced spectrogram for debug. 
    """
    print(args)
    workspace = args.workspace
    tr_snr = args.tr_snr
    te_snr = args.te_snr
    n_concat = args.n_concat
    iter = args.iteration
    n_noise_frame = args.noise_frame
    n_hop = args.n_hop

    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    fs = cfg.sample_rate
    scale = False
    # Load model. 
    model_path = os.path.join(workspace, "models", "%ddb" % int(tr_snr), "md_%diters.h5" % iter)
    model = load_model(model_path)

    # Load scaler. 
    # scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr), "scaler.p")
    # scaler = pickle.load(open(scaler_path, 'rb'))

    # Load test data. 
    feat_dir = os.path.join(workspace, "features", "spectrogram", "test", "%ddb" % int(te_snr))
    names = os.listdir(feat_dir)
    mel_basis = librosa.filters.mel(cfg.sample_rate, cfg.n_window, n_mels=40)
    for (cnt, na) in enumerate(names):
        # Load feature. 
        feat_path = os.path.join(feat_dir, na)
        data = cPickle.load(open(feat_path, 'rb'))
        [mixed_cmplx_x, speech_x, noise_x, alpha, na] = data
        input1_3d, input2, out1, out2 = pp_data.get_input_output_layer(mixed_cmplx_x, speech_x, noise_x, alpha,
                                                                       n_concat, n_noise_frame, n_hop, mel_basis)

        # Predict. 
        pred = model.predict([input1_3d, input2])
        print(cnt, na)
        sys.stdout.flush()

        # Inverse scale. 
        if scale:
            mixed_x = pp_data.inverse_scale_on_2d(mixed_x, scaler)
            speech_x = pp_data.inverse_scale_on_2d(speech_x, scaler)
            pred = pp_data.inverse_scale_on_2d(pred, scaler)

        # post processing
        pred_speech_lps = 1 / 3.0 * (
                pred[0][:, :257] +
                pred[1][:, :257] +
                np.log(np.abs(mixed_cmplx_x) + 1e-08) +
                np.log(pred[1][:, 327:584])
        )

        # Debug plot. 
        if args.visualize:
            out_path = os.path.join(workspace, "figures", "test", "%ddb" % int(te_snr), "%s.all.png" % na)
            pp_data.create_folder(os.path.dirname(out_path))
            fig, axs = plt.subplots(3, 1, sharex=False)
            axs[0].matshow(np.log(np.abs(mixed_cmplx_x.T) + 1e-08), origin='lower', aspect='auto', cmap='jet')
            axs[1].matshow(np.log(speech_x.T + 1e-08), origin='lower', aspect='auto', cmap='jet')
            axs[2].matshow(pred_speech_lps.T, origin='lower', aspect='auto', cmap='jet')
            axs[0].set_title("%ddb mixture log spectrogram" % int(te_snr))
            axs[1].set_title("Clean speech log spectrogram")
            axs[2].set_title("Enhanced speech log spectrogram")
            for j1 in xrange(3):
                axs[j1].xaxis.tick_bottom()
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close('all')
            # plt.show()
            out_path = os.path.join(workspace, "figures", "test", "%ddb" % int(te_snr), "%s.mixture.png" % na)
            display.specshow(np.log(np.abs(mixed_cmplx_x.T) + 1e-08))
            plt.title("%ddb mixture log spectrogram" % int(te_snr))
            plt.savefig(out_path)
            out_path = os.path.join(workspace, "figures", "test", "%ddb" % int(te_snr), "%s.clean.png" % na)
            display.specshow(np.log(speech_x.T + 1e-08))
            plt.title("Clean speech log spectrogram")
            plt.savefig(out_path)
            out_path = os.path.join(workspace, "figures", "test", "%ddb" % int(te_snr), "%s.enh.png" % na)
            display.specshow(pred_speech_lps.T)
            plt.title("Enhanced speech log spectrogram")
            plt.savefig(out_path)
            plt.close('all')

        # Recover enhanced wav.
        pred_sp = np.exp(pred_speech_lps)
        s = recover_wav(pred_sp, mixed_cmplx_x, n_overlap, np.hamming)
        s *= np.sqrt((np.hamming(n_window) ** 2).sum())  # Scaler for compensate the amplitude
        # change after spectrogram and IFFT.

        # Write out enhanced wav. 
        out_path = os.path.join(workspace, "enh_wavs", "test", "%ddb" % int(te_snr), "%s.enh.wav" % na)
        pp_data.create_folder(os.path.dirname(out_path))
        pp_data.write_audio(out_path, s, fs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--tr_snr', type=float, required=True)
    parser_train.add_argument('--te_snr', type=float, required=True)
    parser_train.add_argument('--lr', type=float, required=True)
    parser_train.add_argument('--iter', type=int, required=True)

    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--workspace', type=str, required=True)
    parser_inference.add_argument('--tr_snr', type=float, required=True)
    parser_inference.add_argument('--te_snr', type=float, required=True)
    parser_inference.add_argument('--n_concat', type=int, required=True)
    parser_inference.add_argument('--iteration', type=int, required=True)
    parser_inference.add_argument('--visualize', action='store_true', default=False)
    parser_inference.add_argument('--noise_frame', type=int, required=True)
    parser_inference.add_argument('--n_hop', type=int, required=True)

    parser_calculate_pesq = subparsers.add_parser('calculate_pesq')
    parser_calculate_pesq.add_argument('--workspace', type=str, required=True)
    parser_calculate_pesq.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_pesq.add_argument('--te_snr', type=float, required=True)

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)
    elif args.mode == 'calculate_pesq':
        calculate_pesq(args)
    else:
        raise Exception("Error!")
