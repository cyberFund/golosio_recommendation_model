import pdb
import numpy as np
from nltk.probability import FreqDist
import utils
import sys
from pymongo import MongoClient
import pandas as pd
from gensim import corpora, models
import pdb

WORD_LENGTH_QUANTILE = 10
TEXT_LENGTH_QUANTILE = 66
HIGH_WORD_FREQUENCY_QUANTILE = 99
LOW_WORD_FREQUENCY_QUANTILE = 60
LDA_PARAMETERS = {
  'workers': 10,
  'topics': 100,
  'passes': 5,
  'eta': 0.25
}

def get_posts(url, database):
  client = MongoClient(url)
  db = client[database]
  posts = pd.DataFrame(list(db.comment.find(
    {
      'permlink' : {'$exists' : True},
      'depth': 0,
    }, {
      'permlink': 1,
      'author': 1, 
      'topic' : 1,
      'topic_probability' : 1,
      'parent_permlink': 1,
      'created': 1,
      'json_metadata': 1,
      'body': 1,
    }
  )))
  return utils.preprocess_posts(posts)

def remove_short_words(texts):
  word_lengths = [len(item) for sublist in texts for item in sublist]
  word_length_quantile = np.percentile(np.array(word_lengths), WORD_LENGTH_QUANTILE)
  return [[word for word in text if len(word) >= word_length_quantile] for text in texts]

def remove_short_texts(texts):
  text_lengths = [len(text) for text in texts]
  text_length_quantile = np.percentile(np.array(text_lengths), TEXT_LENGTH_QUANTILE)
  return [text for text in texts if len(text) >= text_length_quantile]

def remove_high_frequent_words(texts):
  dictionary = FreqDist([item for sublist in texts for item in sublist])
  word_frequencies = list(dictionary.values())
  high_word_frequency_quantile = np.percentile(np.array(word_frequencies), HIGH_WORD_FREQUENCY_QUANTILE)
  return [[word for word in text if dictionary[word] < high_word_frequency_quantile] for text in texts]

def remove_low_frequent_words(texts):
  dictionary = FreqDist([item for sublist in texts for item in sublist])
  word_frequencies = list(dictionary.values())
  low_word_frequency_quantile = np.percentile(np.array(word_frequencies), LOW_WORD_FREQUENCY_QUANTILE)
  return [[word for word in text if dictionary[word] >= low_word_frequency_quantile] for text in texts]

def prepare_posts(posts):
  posts = [utils.prepare_post(post) for post in posts]
  posts = remove_short_words(posts)
  posts = remove_high_frequent_words(posts)
  posts = remove_low_frequent_words(posts)
  return remove_short_texts(posts)

def create_dictionary(texts):
  dictionary = corpora.Dictionary(texts)
  dictionary.save('golos-corpora.dict')
  return dictionary

def create_corpus(texts, dictionary):
  corpus = [dictionary.doc2bow(text) for text in texts]
  tfidf = models.TfidfModel(corpus, id2word=dictionary, normalize=True)
  corpora.MmCorpus.serialize('golos-corpora_tfidf.mm', tfidf[corpus])
  return corpora.MmCorpus('golos-corpora_tfidf.mm')

def train_model(corpus, dictionary):
  model = models.LdaMulticore(workers=LDA_PARAMETERS['workers'], corpus=corpus, id2word=dictionary, num_topics=LDA_PARAMETERS['topics'], passes=LDA_PARAMETERS['passes'], eta=LDA_PARAMETERS['eta'])
  return model

def create_model(texts):
  dictionary = create_dictionary(texts)
  corpus = create_corpus(texts, dictionary)
  model = train_model(corpus, dictionary)
  model.save('golos.lda_model')
  return model, dictionary

def run_lda(database_url, database_name):
  print("Get posts...")
  posts = get_posts(database_url, database_name)
  print("Prepare posts...")
  texts = prepare_posts(posts["body"])
  print("Prepare model...")
  model, dictionary = create_model(texts)
  print("Save topics...")
  utils.save_topics(database_url, database_name, posts, model, dictionary)

if (__name__ == "__main__"):
  run_lda(sys.argv[1], sys.argv[2])