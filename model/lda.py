import pdb
import numpy as np
from nltk.probability import FreqDist
import utils
import sys
from pymongo import MongoClient
import pandas as pd
from gensim import corpora, models
import pdb
from tqdm import *
import datetime as dt

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

HOURS_LIMIT = 365 * 24 # Time window for analyzed posts

def get_posts(url, database):
  """
    Function to get last posts from mongo database
  """
  date = dt.datetime.now() - dt.timedelta(hours=HOURS_LIMIT)
  client = MongoClient(url)
  db = client[database]
  posts = pd.DataFrame(list(db.comment.find(
    {
      'permlink' : {'$exists' : True},
      'depth': 0,
      'created': {'$gte': date}
    }, {
      'permlink': 1,
      'author': 1, 
      'topic' : 1,
      'topic_probability' : 1,
      'parent_permlink': 1,
      'created': 1,
      'json_metadata': 1,
      'body': 1,
      'prepared_body': 1
    }
  )))
  return utils.preprocess_posts(posts)

def remove_short_words(texts):
  """
    Function to remove short words from texts
  """
  utils.log("LDA", "Find length of words...")
  word_lengths = [len(item) for sublist in tqdm(texts) for item in sublist]
  word_length_quantile = np.percentile(np.array(word_lengths), WORD_LENGTH_QUANTILE)
  utils.log("LDA", "Remove short words...")
  return [[word for word in text if len(word) >= word_length_quantile] for text in tqdm(texts)]

def remove_short_texts(texts):
  """
    Function to remove short texts from a corpus
  """
  utils.log("LDA", "Find length of texts...")
  text_lengths = [len(text) for text in tqdm(texts)]
  text_length_quantile = np.percentile(np.array(text_lengths), TEXT_LENGTH_QUANTILE)
  utils.log("LDA", "Remove short texts...")
  return [text for text in texts if len(text) >= text_length_quantile]

def remove_high_frequent_words(texts):
  """
    Function to remove high frequent words from texts
  """
  utils.log("LDA", "Remove high frequent words...")
  dictionary = FreqDist([item for sublist in texts for item in sublist])
  word_frequencies = list(dictionary.values())
  high_word_frequency_quantile = np.percentile(np.array(word_frequencies), HIGH_WORD_FREQUENCY_QUANTILE)
  return [[word for word in text if dictionary[word] < high_word_frequency_quantile] for text in tqdm(texts)]

def remove_low_frequent_words(texts):
  """
    Function to remove low frequent words from texts
  """
  utils.log("LDA", "Remove low frequent words...")
  dictionary = FreqDist([item for sublist in texts for item in sublist])
  word_frequencies = list(dictionary.values())
  low_word_frequency_quantile = np.percentile(np.array(word_frequencies), LOW_WORD_FREQUENCY_QUANTILE)
  return [[word for word in text if dictionary[word] >= low_word_frequency_quantile] for text in tqdm(texts)]

def prepare_posts(posts):
  """
    Function to prepare each post and to prepare texts for LDA algorithm
  """
  posts = [utils.prepare_post(posts.loc[index]) for index in tqdm(posts.index)]
  usable_posts = remove_short_words(posts)
  usable_posts = remove_high_frequent_words(usable_posts)
  usable_posts = remove_low_frequent_words(usable_posts)
  usable_posts = remove_short_texts(usable_posts)
  return posts, usable_posts

def create_dictionary(texts):
  """
    Function to create dictionary from texts
  """
  dictionary = corpora.Dictionary(texts)
  dictionary.save('golos-corpora.dict')
  return dictionary

def create_corpus(texts, dictionary):
  """
    Function to create corpus from texts and created dictionary
  """
  corpus = [dictionary.doc2bow(text) for text in tqdm(texts)]
  tfidf = models.TfidfModel(corpus, id2word=dictionary, normalize=True)
  corpora.MmCorpus.serialize('golos-corpora_tfidf.mm', tfidf[corpus])
  return corpora.MmCorpus('golos-corpora_tfidf.mm')

def train_model(corpus, dictionary):
  """
    Function to train LDA model on prepared corpus
  """
  model = models.LdaMulticore(workers=LDA_PARAMETERS['workers'], corpus=corpus, id2word=dictionary, num_topics=LDA_PARAMETERS['topics'], passes=LDA_PARAMETERS['passes'], eta=LDA_PARAMETERS['eta'])
  return model

def create_model(texts):
  """
    Function to create and to save LDA model
  """
  dictionary = create_dictionary(texts)
  corpus = create_corpus(texts, dictionary)
  model = train_model(corpus, dictionary)
  model.save('golos.lda_model')
  return model, dictionary

@utils.error_log("LDA")
def run_lda(database_url, database_name):
  """
    Function to run LDA process:
    - Get all posts from mongo
    - Prepare post bodies
    - Create LDA model
    - Find and save LDA topics for each model
  """
  utils.log("LDA", "Get posts...")
  posts = get_posts(database_url, database_name)
  utils.log("LDA", "Prepare posts...")
  texts, usable_texts = prepare_posts(posts)
  utils.log("LDA", "Prepare model...")
  model, dictionary = create_model(usable_texts)
  utils.log("LDA", "Save topics...")
  utils.wait_and_lock_mutex(database_url, database_name, "lda")
  utils.save_topics(database_url, database_name, posts, texts, model, dictionary)
  utils.unlock_mutex(database_url, database_name, "lda")

if (__name__ == "__main__"):
  run_lda(sys.argv[1], sys.argv[2])
