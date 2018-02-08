from golosio_recommendation_model.model import utils
import sys
from gensim import models, corpora
import pdb
from golosio_recommendation_model.model.train.doc2vec import prepare_posts
from golosio_recommendation_model.config import config
from pymongo import MongoClient
from tqdm import *

DOC2VEC_STEPS = 2500
DOC2VEC_ALPHA = 0.03

def get_posts(url, database):
  events = utils.get_events(url, database)
  posts = utils.get_posts(url, database, events, {
    'inferred_vector' : {'$exists' : False}
  })
  return utils.preprocess_posts(posts)

def save_document_vectors(url, database, posts, texts, model):
  """
    Function to save Doc2Vec vectors for each post to mongo database
  """
  client = MongoClient(url)
  db = client[database]
  posts["prepared_body"] = texts
  for index in tqdm(posts.index):
    post = posts.loc[index]
    inferred_vector = model.infer_vector(post["prepared_body"], steps=DOC2VEC_STEPS, alpha=DOC2VEC_ALPHA)
    db.comment.update_one({'_id': post["post_permlink"][1:]}, {'$set': {'inferred_vector': inferred_vector.tolist()}})  

@utils.error_log("Doc2Vec predict")
def predict_doc2vec():
  database_url = config['database_url']
  database_name = config['database_name']
  utils.log("Doc2Vec predict", "Restore model...")
  model = models.doc2vec.Doc2Vec.load(config['model_path'] + 'golos.doc2vec_model')

  while True:
    utils.wait_between_iterations()
    utils.log("Doc2Vec predict", "Get posts...")
    posts = get_posts(database_url, database_name)
    if posts.shape[0] > 0:
      utils.log("Doc2Vec predict", "Prepare posts...")
      texts, usable_texts = prepare_posts(posts)
      utils.log("Doc2Vec predict", "Save inferred vectors...")
      save_document_vectors(database_url, database_name, posts, texts, model)
