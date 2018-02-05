from model import utils
import sys
from gensim import models, corpora
import pdb
from model.train.doc2vec import prepare_posts, save_document_vectors

def get_posts(url, database):
  events = utils.get_events(url, database)
  posts = utils.get_posts(url, database, events, {
    'inferred_vector' : {'$exists' : False}
  })
  return utils.preprocess_posts(posts)

def run_doc2vec(database_url, database_name):
  utils.log("Doc2Vec predict", "Get posts...")
  posts = get_posts(database_url, database_name)
  utils.log("Doc2Vec predict", "Prepare posts...")
  texts, usable_texts = prepare_posts(posts)
  utils.log("Doc2Vec predict", "Restore model...")
  model = models.doc2vec.Doc2Vec.load('golos.doc2vec_model')
  utils.log("Doc2Vec predict", "Save inferred vectors...")
  save_document_vectors(database_url, database_name, posts, texts, model)

if (__name__ == "__main__"):
  run_doc2vec(sys.argv[1], sys.argv[2])