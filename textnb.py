import string
import numpy as np
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter


def get_word_counts(words):
    """This function returns a dictionary of word counts."""
    return dict(Counter(words))


def normalize_text(word_tokens, lemmas_words=False):
    """This function uses Snowball stemmer or WordNet Lemmatizer to normalize words.
    """
    normalized_words = []
    # Lemmatize words
    if lemmas_words:
        wordnet_lemmatizer = WordNetLemmatizer()
        for word in word_tokens:
            if len(word) > 1:
                normalized_words.append(wordnet_lemmatizer.lemmatize(word))
    # Stem words
    else:
        snowball = SnowballStemmer('english')
        for word in word_tokens:
            if len(word) > 1:
                normalized_words.append(snowball.stem(word))
    return normalized_words


def tokenize_text(text, remove_stopwords=True):
    """This function will tokenize text and remove stopwords.
    """
    word_tokens = []
    # Sentence tokenization
    sentences_nltk = sent_tokenize(text)
    for sentence in sentences_nltk:
        # Remove punctuation
        sentence = sentence.replace('-', ' ').translate(str.maketrans('', '', string.punctuation)).lower()
        # Word tokenization  - leave alphabetic characters only
        tokens = [w for w in word_tokenize(sentence) if w.isalpha() and len(w) > 1]  # Remove single characters

        # Remove stopwords
        if remove_stopwords:
            # A more comprehensive self-defined list of could be defined like for words like 'BBC News' etc.
            no_stops = [t for t in tokens if t not in stopwords.words('english')]
            word_tokens.extend(no_stops)
        else:
            word_tokens.extend(tokens)
    return word_tokens


class TextNaiveBayes:
    """Implementation of Naive Bayes classifier for binary text classification."""

    def __init__(self):
        self.vocab = set()  # All distinct word tokens
        self.word_counts = {}  # Word counts for each class
        self.log_class_priors = {}  # Log of Prior probabilities for each class
        self.num_articles = {}  # No of docs

    def fit(self, X_train, y_train):
        """This function calculates log class prior probabilities from the training data."""

        n = len(X_train)
        self.num_articles['is_target'] = sum(1 for label in y_train if label)
        self.num_articles['not_target'] = sum(1 for label in y_train if not label)
        self.log_class_priors['is_target'] = np.log(self.num_articles['is_target'] / n)
        self.log_class_priors['not_target'] = np.log(self.num_articles['not_target'] / n)
        self.word_counts['is_target'] = {}
        self.word_counts['not_target'] = {}

        for x, y in zip(X_train, y_train):
            c = 'is_target' if y else 'not_target'
            tokenized_text = tokenize_text(x)
            normalized_text = normalize_text(tokenized_text)
            counts = get_word_counts(normalized_text)

            for word, count in counts.items():
                if word not in self.vocab:
                    self.vocab.add(word)
                if word not in self.word_counts[c]:
                    self.word_counts[c][word] = 0.0

                self.word_counts[c][word] += count

    def predict(self, X_test):
        """This function uses Naive Bayes calculation to return a list of predicted class labels."""
        result = []
        for x in X_test:
            tokenized_text = tokenize_text(x)
            normalized_text = normalize_text(tokenized_text)
            counts = get_word_counts(normalized_text)
            is_target_score = 0
            not_target_score = 0

            for word, _ in counts.items():
                # Add Laplace smoothing for test words that are not present in training set
                if word not in self.vocab:
                    continue

                # Log conditional probability for word given class
                log_w_given_is_target = np.log((self.word_counts['is_target'].get(word, 0) + 1) / (self.num_articles['is_target'] + len(self.vocab)) )
                log_w_given_not_target = np.log((self.word_counts['not_target'].get(word, 0) + 1) / (self.num_articles['not_target'] + len(self.vocab)) )

                # Sum all the log conditional probabilities of all the words
                is_target_score += log_w_given_is_target
                not_target_score += log_w_given_not_target

            # Final score (not a true probability)
            is_target_score += self.log_class_priors['is_target']
            not_target_score += self.log_class_priors['not_target']

            # Select class with the highest score
            if is_target_score > not_target_score:
                result.append(1)
            else:
                result.append(0)
        return result

    def score(self, X_test, y_test):
        """This function return accuracy score of the test data."""
        y_predicted = self.predict(X_test)
        return accuracy_score(y_test, y_predicted)
