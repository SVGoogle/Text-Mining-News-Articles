# Text Mining. News Articles Data

## Dataset
- News Article data is described here: <http://sciride.org/news.html>
- The processed articles are available here (~16 GB, download completed in May 2021): [Articles Dataset](https://news-mine.s3.eu-west-2.amazonaws.com/processed.tar.gz)
- With the relevant documentation here: <http://sciride.org/news.html#datacontent>

## Implementing Naive Bayes Classifier for Text Classification from Scratch

The classifier is implemented as a Python class object called TextNaiveBayes . 
The class objects implement the following text processing pipeline and methods:
1. Tokenization of the data (from text to sentences to words), removing punctuation and English stopwords
2. Normalization of the tokenized text via stemming (or lemmatization)
3. Creating a vocabulary of distinct word tokens
4. Counting word tokens in each class (is_covid, not_covid)
5. Fitting the training data
6. Predicting labels on the test data
7. Estimating the performance of the classifier using accuracy score metric

>### Model training results
>
>- The model is trained on 80% data
>- Text Naive Bayes classifier achieved 96% accuracy on the remaining 20% test data
>

## Analysis of COVID-19 Related News Article Proportions
In this section the previously trained Naive Bayes classifier is used to predict if the news article is related to COVID-19 topic.

The figure below shows the proportions of all articles published in 2020 compared to BBC News outlet.
![Proportions 2020](./Results/covid19_proportion_2020.png)
It can be seen that BBC News has almost the same proportions with ~21% of articles related to COVID-19.

The figure below shows the monthly proportions of articles. Data from December month is not complete.
![Monthly Proportions](./Results/covid19_proportion_monthly_2020.png)

## Named Entitity Recognition
In this section the most commonly mentioned Named Entities with respect to COVID-19 are extracted using the statistical model from SpaCy library.
