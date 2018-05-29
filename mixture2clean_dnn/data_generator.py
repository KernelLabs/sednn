import numpy as np

class DataGenerator(object):
    def __init__(self, batch_size, type,max_len, te_max_iter=None):
        assert type in ['train', 'test']
        self._batch_size_ = batch_size
        self._type_ = type
        self._te_max_iter_ = te_max_iter
        self.max_len = max_len
        
    def generate(self, xs, ys, utts):
        x1 = xs[0]
        x2 = xs[1]
        adapt_utt = xs[2]

        y1 = ys[0]
        y2 = ys[1]
        batch_size = self._batch_size_
        n_samples = len(x1)

        index = np.arange(n_samples)
        np.random.shuffle(index)

        iter = 0
        epoch = 0
        pointer = 0
        while True:
            if (self._type_ == 'test') and (self._te_max_iter_ is not None):
                if iter == self._te_max_iter_:
                    break
            iter += 1
            if pointer >= n_samples:
                epoch += 1
                if (self._type_) == 'test' and (epoch == 1):
                    break
                pointer = 0
                np.random.shuffle(index)

            batch_idx = index[pointer : min(pointer + batch_size, n_samples)]
            pointer += batch_size
            utt_feature=np.zeros((len(batch_idx),self.max_len*257))
            for idx,utt in enumerate(adapt_utt[batch_idx]):
                utt_feature[idx]=np.reshape(np.pad(utts[utt],((0,self.max_len-len(utts[utt])),(0,0)),'constant'),-1)
            yield [x1[batch_idx],x2[batch_idx], np.array(utt_feature)], [y1[batch_idx],y2[batch_idx]]