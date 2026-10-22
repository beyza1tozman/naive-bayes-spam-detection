# Spam Detection With Naive Bayes

Spam email classifier with Python, numpy and scikit-learn. It compares all five Naive Bayes models in sklearn on the same data, and also has multinomial Naive Bayes written from scratch with numpy to see how it works inside.

## Dataset

Ling-Spam, the preprocessed subset (stop words removed, lemmatized):
https://www.dropbox.com/s/yjiplngoa430rid/ling-spam.zip

The data is not in the repo. Unzip it so the folders are next to the notebooks:

```
spam-detection/
  spam_classifier.ipynb
  train-mails/   702 mails, 351 spam
  test-mails/    260 mails, 130 spam
```

Spam mails are the files that start with `spmsg`, the rest are normal mails (ham).

The original full corpus is here if needed: https://www2.aueb.gr/users/ion/data/lingspam_public.tar.gz

## Setup

```
git clone https://github.com/beyza1tozman/spam-detection.git
cd spam-detection
python -m venv venv
venv\Scripts\activate
pip install numpy pandas matplotlib scikit-learn jupyter
```

Then open a notebook, pick the venv as kernel and run all.

## Files

- `spam_classifier.ipynb` trains sklearn's `MultinomialNB`, then compares it with `GaussianNB`, `ComplementNB`, `BernoulliNB` and `CategoricalNB`
- `naive_bayes_from_scratch.ipynb` is multinomial Naive Bayes written with numpy and checked against sklearn
- `data_prep.py` has the shared functions both notebooks use: `read_mails`, `build_dictionary`, `to_vectors`
- `exploration.ipynb` has my first tries before the working code, just to remember how I got there

## How It Works

1. Read all mails and get labels from the file names
2. Create a dictionary of the 2000 most frequent words in the train set. Drop punctuation, numbers and single characters
3. Transform each mail into a word count vector (702 x 2000 for train, 260 x 2000 for test)
4. Train the models on train and check them on the test set

## Comparison Of sklearn Naive Bayes Models

The five models differ in how they look at the word counts:

- `GaussianNB` treats every feature as a normal distribution, made for continuous data
- `MultinomialNB` uses the word counts, the classic one for text
- `ComplementNB` is like multinomial but learns from the other class, meant for imbalanced data
- `BernoulliNB` only checks if a word is in the mail or not
- `CategoricalNB` treats every count value as its own category

Results on the test set (precision, recall and f1 are for the spam class):

| model | accuracy | precision | recall | f1 | roc auc |
|---|---|---|---|---|---|
| Gaussian | 0.973 | 0.992 | 0.954 | 0.973 | 0.973 |
| Multinomial | 0.965 | 1.000 | 0.931 | 0.964 | 0.998 |
| Complement | 0.965 | 1.000 | 0.931 | 0.964 | 0.998 |
| Bernoulli | 0.758 | 0.986 | 0.523 | 0.683 | 0.990 |
| Categorical | 0.719 | 0.967 | 0.454 | 0.618 | 0.983 |

Gaussian has the best accuracy and f1, but it sends 1 real mail to spam. Multinomial and Complement give the same result because the classes are balanced. They never mark a real mail as spam, 9 spam mails get through as ham, and they have the best roc auc. Bernoulli and Categorical miss about half of the spam because they lose the information that a higher count means more of a word.

CategoricalNB breaks if the test set has a count it never saw in train, so test counts are cut to the max count of that word in train.

## Naive Bayes From Scratch

`MyMultinomialNB` in `naive_bayes_from_scratch.ipynb` does the same thing as sklearn's `MultinomialNB`:

- class prior is the share of mails in each class
- word probabilities are the word counts per class with laplace smoothing
- prediction uses log probabilities so 2000 small numbers dont underflow to 0, which makes it one matrix multiplication

It gives exactly the same priors, word probabilities and predictions as sklearn, so the accuracy is the same, 96.5%. The notebook also walks through the scores of one mail and lists the most spammy and most hammy words.
