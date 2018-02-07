from golosio_recommendation_model.model import utils
import sys
from gensim import models, corpora
import pdb
from golosio_recommendation_model.model.train.doc2vec import prepare_posts, save_document_vectors
from golosio_recommendation_model.config import config

def get_posts(url, database):
  events = utils.get_events(url, database)
  posts = utils.get_posts(url, database, events, {
    'inferred_vector' : {'$exists' : False}
  })
  return utils.preprocess_posts(posts)

@utils.error_log("Doc2Vec predict")
def run_doc2vec():
  database_url = config['database_url']
  database_name = config['database_name']
  utils.log("Doc2Vec predict", "Restore model...")
  utils.wait_for_file(config['model_path'] + 'golos.doc2vec_model')
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