import pdb
import numpy as np
from nltk.probability import FreqDist
import utils
import sys
from pymongo import MongoClient
import pandas as pd
from gensim import corpora, models
import pdb
import datetime as dt
from tqdm import *

WORD_LENGTH_QUANTILE = 10
TEXT_LENGTH_QUANTILE = 66
HIGH_WORD_FREQUENCY_QUANTILE = 99
LOW_WORD_FREQUENCY_QUANTILE = 60
NUMBER_OF_RECOMMENDATIONS = 10
DOC2VEC_PARAMETERS = {
  'size': 100,
  'window': 8,
  'min_count': 5,
  'workers': 13
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
    }
  )))
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
  posts = [utils.prepare_post(post) for post in tqdm(posts)]
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
  model.save('golos.doc2vec_model')
  return model

def save_document_vectors(url, database, posts, texts, model):
  """
    Function to save Doc2Vec vectors for each post to mongo database
  """
  client = MongoClient(url)
  db = client[database]
  posts["prepared_body"] = texts
  for index in tqdm(posts.index):
    post = posts.loc[index]
    inferred_vector = model.infer_vector(post["prepared_body"])
    db.comment.update_one({'_id': post["post_permlink"][1:]}, {'$set': {'inferred_vector': inferred_vector.tolist()}})  

def run_doc2vec(database_url, database_name):
  """
    Function to run Doc2Vec process:
    - Get all posts from mongo
    - Prepare post bodies
    - Create Doc2Vec model
    - Find and save Doc2Vec vectors for each model
  """
  utils.log("Doc2Vec", "Get posts...")
  posts = get_posts(database_url, database_name)
  utils.log("Doc2Vec", "Prepare posts...")
  texts, usable_texts = prepare_posts(posts["body"])
  utils.log("Doc2Vec", "Prepare model...")
  model = create_model(usable_texts)
  utils.log("Doc2Vec", "Save vectors...")
  utils.wait_and_lock_mutex(database_url, database_name, "doc2vec")
  save_document_vectors(database_url, database_name, posts, texts, model)
  utils.unlock_mutex(database_url, database_name, "doc2vec")

if (__name__ == "__main__"):
  run_doc2vec(sys.argv[1], sys.argv[2])
