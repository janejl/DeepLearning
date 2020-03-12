import os
import glob
import numpy as np
import tensorflow_datasets as tfds


class SmallImdbDataset(object):
    """This is a small dataset for debug purpose."""
    def get_train_test(self):
        sentences = np.asarray([['one', 'of', 'the', 'best', 'comedy', 'series', 'to', 'ever', 'come', 'out'],
                     ['this', 'may', 'be', 'the', 'worst', 'show', 'i', 've', 'ever', 'seen']])
        labels = np.asarray([1, 0], dtype=np.int8)
        return (sentences, labels), (sentences, labels)


class ImdbDataset(object):
    """
    Pre-processed IMDB dataset. All the sentences are tokenized with tfds.features.text.Tokenizer and lowercased.
    Original dataset can be downloaded from https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz.
    """

    def __init__(self):
        data_dir = os.path.dirname(os.path.realpath(__file__))
        self.imdb_path = os.path.join(data_dir, "imdb/imdb.npz")

    def get_train_test(self):
        """
        Load train and test data from imdb.npz
        Return:
            Tuple of ((x_train, y_train), (x_test, y_test)). Both train and test sets contain 25000 samples.
            - x_train, x_test: input train and test tokens;
            - y_train, y_test: output train and test labels. 0 represents negative review and 1 positive review.
        """
        print("Loading data from %s.." % self.imdb_path)
        train_test = np.load(self.imdb_path, allow_pickle=True)
        return (train_test["x_train"], train_test["y_train"]), (train_test["x_test"], train_test["y_test"])


if __name__ == '__main__':

    imdb_dataset = ImdbDataset()
    #
    # # tokenizer = tfds.features.text.Tokenizer()
    # # a = "I'm fine."
    # # print(tokenizer.tokenize(a))
    #
    # # x, y = imdb_dataset.convert_to_numpy()
    # # print(x)
    # # print(y[1])
    #
    (x_train, y_train), (x_test, y_test) = imdb_dataset.get_train_test()
    #
    print(x_train.shape, x_test.shape)
    print(x_train[:10])
    print(y_train[:10])
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)