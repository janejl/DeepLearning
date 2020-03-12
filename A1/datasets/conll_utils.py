import os
import numpy as np


class ConllDataset(object):
    """Load data from datasets/conll2003
        - train.txt: training data
        - dev.txt: validation data
        - test.masked.txt: masked test data. All the label in the data are masked as "O".
    Note: pad will be added to the sequences util their lengths equals to the max_length.
        The index for pad is 0, for both input and output sequences.
    """

    def __init__(self):
        # self.data_dir = data_dir
        dataset_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(dataset_dir, "conll2003")

        # Load vocabulary
        self.vocab = self.load_vocab(data_dir)
        self.vocab_size = len(self.vocab)
        # Label to index
        self.label2index = {"<pad>": 0, "O": 1, "PER": 2, "ORG": 3, "LOC": 4, "MISC": 5}
        # Index to label
        self.index2label = dict([(v, k) for k, v in self.label2index.items()])

        # Read tokens and labels from txt files
        raw_trainset = self._read_conll(os.path.join(data_dir, "train.txt"))
        raw_devset = self._read_conll(os.path.join(data_dir, "dev.txt"))
        raw_testset = self._read_conll(os.path.join(data_dir, "test.masked.txt"))
        allset = raw_trainset + raw_devset + raw_testset
        # Maximum length of the datasets
        self.max_length = max([len(tok_lbl[0]) for tok_lbl in allset])

        # Numpy array for training & testing
        self.x_train, self.y_train = self.convert_and_pad_dataset(raw_trainset)
        self.x_dev, self.y_dev = self.convert_and_pad_dataset(raw_devset)
        self.x_test, _ = self.convert_and_pad_dataset(raw_testset)
        # Test set tokens, for the purpose of prediction
        self.testset = raw_testset

    def convert_and_pad_single_sequence(self, tok_lbl):
        tok, lbl = tok_lbl
        num_tok, num_lbl = len(tok), len(lbl)
        assert num_tok == num_lbl

        tok_idx = [self.vocab[w] for w in tok]
        lbl_idx = [self.label2index[l] for l in lbl]

        if num_tok < self.max_length:
            tok_idx.extend([0] * (self.max_length-num_tok))
            lbl_idx.extend([0] * (self.max_length-num_tok))

        tok_idx = np.asarray(tok_idx, dtype=np.int32)
        lbl_idx = np.asarray(lbl_idx, dtype=np.int32)
        return tok_idx, lbl_idx

    def convert_and_pad_dataset(self, dataset):
        num_samples = len(dataset)
        dataset_tokens = np.zeros([num_samples, self.max_length], dtype=np.int32)
        dataset_labels = np.zeros([num_samples, self.max_length], dtype=np.int32)
        for i, tok_lbl in enumerate(dataset):
            dataset_tokens[i], dataset_labels[i] = self.convert_and_pad_single_sequence(tok_lbl)
        return dataset_tokens, dataset_labels

    @staticmethod
    def _read_conll(fpath):
        ret = []
        current_toks, current_lbls = [], []
        with open(fpath, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if len(line) == 0:
                    if len(current_toks) > 0:
                        assert len(current_toks) == len(current_lbls)
                        ret.append((current_toks, current_lbls))
                    current_toks, current_lbls = [], []
                else:
                    tok, lbl = line.split()
                    current_toks.append(tok)
                    current_lbls.append(lbl)
            if len(current_toks) > 0:
                assert len(current_toks) == len(current_lbls)
                ret.append((current_toks, current_lbls))
        return ret

    def load_vocab(self, data_dir):
        vocab_dict = {}
        with open(os.path.join(data_dir, "vocab.txt"), "r", encoding="utf-8") as fp:
            for i, line in enumerate(fp):
                token = line.strip()
                vocab_dict[token] = i
        return vocab_dict


if __name__ == '__main__':

    dataset = ConllDataset()

    print(dataset.label2index)
    print(dataset.index2label)
    print(dataset.vocab)
    print(dataset.x_train.shape)
    print(dataset.x_dev.shape)
    print(dataset.x_test.shape)
