import pdb
import numpy as np
from nltk.probability import FreqDist
from golosio_recommendation_model.model import utils
import sys
from pymongo import MongoClient
import pandas as pd
from gensim import corpora, models
import pdb
import datetime as dt
from tqdm import *
from golosio_recommendation_model.config import config
from golosio_recommendation_model.model.predict.doc2vec import predict_doc2vec
from golosio_recommendation_model.daemonize import daemonize

WORD_LENGTH_QUANTILE = 10
TEXT_LENGTH_QUANTILE = 66
HIGH_WORD_FREQUENCY_QUANTILE = 99.5
LOW_WORD_FREQUENCY_QUANTILE = 60
NUMBER_OF_RECOMMENDATIONS = 10
DOC2VEC_PARAMETERS = {
  'size': 300,
  'window': 20,
  'min_count': 5,
  'workers': 13
}
DOC2VEC_STEPS = 2500
DOC2VEC_ALPHA = 0.03

def get_posts(url, database):
  events = utils.get_events(url, database)
  posts = utils.get_posts(url, database, events)
  return utils.preprocess_posts(posts)

def remove_short_words(texts):
  """
    Function to remove short words from texts
  """
  utils.log("Doc2Vec", "Find length of words...")
  word_lengths = [len(item) for sublist in tqdm(texts) for item in sublist]
  word_length_quantile = np.percentile(np.array(word_lengths), WORD_LENGTH_QUANTILE)
  utils.log("Doc2Vec", "Remove short words...")
  return [[word for word in text if len(word) >= word_length_quantile] for text in tqdm(texts)]

def remove_short_texts(texts):
  """
    Function to remove short texts from a corpus
  """
  utils.log("Doc2Vec", "Find length of texts...")
  text_lengths = [len(text) for text in tqdm(texts)]
  text_length_quantile = np.percentile(np.array(text_lengths), TEXT_LENGTH_QUANTILE)
  utils.log("Doc2Vec", "Remove short texts...")
  return [text for text in texts if len(text) >= text_length_quantile]

def remove_high_frequent_words(texts):
  """
    Function to remove high frequent words from texts
  """
  utils.log("Doc2Vec", "Remove high frequent words...")
  dictionary = FreqDist([item for sublist in texts for item in sublist])
  word_frequencies = list(dictionary.values())
  high_word_frequency_quantile = np.percentile(np.array(word_frequencies), HIGH_WORD_FREQUENCY_QUANTILE)
  return [[word for word in text if dictionary[word] < high_word_frequency_quantile] for text in tqdm(texts)]

def remove_low_frequent_words(texts):
  """
    Function to remove low frequent words from texts
  """
  utils.log("Doc2Vec", "Remove low frequent words...")
  dictionary = FreqDist([item for sublist in texts for item in sublist])
  word_frequencies = list(dictionary.values())
  low_word_frequency_quantile = np.percentile(np.array(word_frequencies), LOW_WORD_FREQUENCY_QUANTILE)
  return [[word for word in text if dictionary[word] >= low_word_frequency_quantile] for text in tqdm(texts)]

def prepare_posts(posts):
  """
    Function to prepare each post and to prepare texts for Doc2Vec algorithm
  """
  posts = [utils.prepare_post(posts.loc[index]) for index in tqdm(posts.index)]
  posts = remove_short_words(posts)
  posts = remove_high_frequent_words(posts)
  posts = remove_low_frequent_words(posts)
  usable_posts = remove_short_texts(posts)
  return posts, usable_posts

def create_corpus(texts):
  """
    Function to convert texts to a list of tagged documents
  """
  return [models.doc2vec.TaggedDocument(text, [index]) for index, text in enumerate(texts)]

def train_model(corpus):
  """
    Function to train Doc2Vec model on prepared corpus
  """
  model = models.doc2vec.Doc2Vec(**DOC2VEC_PARAMETERS)
  model.build_vocab(corpus)
  model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)
  return model

def create_model(texts):
  """
    Function to create and to save doc2vec model
  """
  corpus = create_corpus(texts)
  model = train_model(corpus)
  return model

def unset_inferred_vectors(url, database):
  client = MongoClient(url)
  db = client[database]
  db.comment.update_many({}, {'$unset': {'inferred_vector': ""}})

@utils.error_log("Doc2Vec train")
def train_doc2vec():
  """
    Function to run Doc2Vec process:
    - Get all posts from mongo
    - Prepare post bodies
    - Create Doc2Vec model
    - Find and save Doc2Vec vectors for each model
  """
  database_url = config['database_url']
  database_name = config['database_name']

  utils.log("Doc2Vec train", "Get posts...")
  posts = get_posts(database_url, database_name)
  utils.log("Doc2Vec train", "Prepare posts...")
  texts, usable_texts = prepare_posts(posts)
  utils.log("Doc2Vec train", "Prepare model...")
  model = create_model(usable_texts)
  daemonize(predict_doc2vec, "stop")
  utils.log("Doc2Vec train", "Save model...")
  model.save(config['model_path'] + 'golos.doc2vec_model')
  unset_inferred_vectors(database_url, database_name)
  daemonize(predict_doc2vec, "start")