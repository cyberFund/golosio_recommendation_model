import utils
import sys
from pymongo import MongoClient
import pandas as pd
from gensim import models, corpora
import pdb

def get_new_posts(url, database):
  posts = pd.DataFrame()
  client = MongoClient(url)
  db = client[database]
  posts = pd.DataFrame(list(db.comment.find(
    {
      'permlink' : {'$exists' : True},
      'topic' : {'$exists' : False},
      'depth': 0,
    }, {
      'permlink': 1,
      'author': 1, 
      'topic' : 1,
      'topic_probability' : 1,
      'parent_permlink': 1,
      'created': 1,
      'json_metadata': 1,
      'body': 1
    }
  )))
  return utils.preprocess_posts(posts)

def run_lda(database_url, database_name):
  print("Getting new posts...")
  posts = get_new_posts(database_url, database_name)
  dictionary = corpora.Dictionary.load('golos-corpora.dict')
  model = models.LdaMulticore.load('golos.lda_model')
  utils.save_topics(database_url, database_name, posts, model, dictionary)

if (__name__ == "__main__"):
  run_lda(sys.argv[1], sys.argv[2])