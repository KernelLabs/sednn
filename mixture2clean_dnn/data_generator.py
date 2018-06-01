import numpy as np


class DataGenerator(object):
    def __init__(self, batch_size, type, max_len, te_max_iter=None):
        assert type in ['train', 'test']
        self._batch_size_ = batch_size
        self._type_ = type
        self._te_max_iter_ = te_max_iter
        self.max_len = max_len

    def generate(self, xs, ys, utts):
        """
        batch data generator
        :param xs: input data
        :param ys: target(label) data
        :param utts: list of speaker name
        :return:
        """
        
    def generate(self, xs, ys):
            utt_feature = np.zeros((len(batch_idx), self.max_len * 257))
            for idx, utt in enumerate(adapt_utt[batch_idx]):
                utt_feature[idx] = np.reshape(
                    np.pad(utts[utt], ((0, self.max_len - len(utts[utt])), (0, 0)), 'constant'), -1)
            yield [x1[batch_idx], x2[batch_idx], np.array(utt_feature)], [y1[batch_idx], y2[batch_idx]]
