from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class BOW():
    """
    Bag-of-Words representation

    "num_words" covariates
    each covariate is the frequency of one 
    word in self.vocab. The words are selected
    by frequency. New words are discarted. Example

    b = BOW({"num_words": 3})
    c = ["are are are are are is is is you you no"]
    b.fit(c)
    b.transform(c).toarray()

    - > array([[5, 3, 2]])

    b.vocab

    - > {'are': 0, 'is': 1, 'you': 2}

    b.transform(["i love you"]).toarray()

    - > array([[0, 0, 1]])
    """

    def __init__(self, hyperparams):
        """
        :param hyperparams: hyperparams dict
                            variables:
                             - "num_words", str

        :type hyperparams: dict
        """
        self.num_words = hyperparams["num_words"]
        self.vectorizer = CountVectorizer(max_features=self.num_words)

    def fit(self, corpus):
        """
        :param corpus: corpus
        :type corpus: [str]
        """
        self.vectorizer.fit(corpus)
        self.vocab = self.vectorizer.vocabulary_

    def transform(self, corpus):
        """
        :param corpus: corpus
        :type corpus: [str]
        :return: prediction
        :rtype: np.array

        """
        return self.vectorizer.transform(corpus)


class Tfidf():
    """
    Convert a collection of raw documents to a matrix of TF-IDF features.

    Equivalent to :class:`CountVectorizer` followed by
    :class:`TfidfTransformer`
    """

    def __init__(self, hyperparams):
        """
        :param hyperparams: hyperparams dict

        :type hyperparams: dict
        """
        self.max_features = hyperparams["max_features"]
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)

    def fit(self, corpus):
        """
        :param corpus: corpus
        :type corpus: [str]
        """
        self.vectorizer = self.vectorizer.fit(corpus)
        self.vocab = self.vectorizer.vocabulary_

    def transform(self, corpus):
        """
        :param corpus: corpus
        :type corpus: [str]
        :return: prediction
        :rtype: np.array

        """
        return self.vectorizer.transform(corpus)
