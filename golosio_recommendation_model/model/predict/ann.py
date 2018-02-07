from golosio_recommendation_model.model import utils
import sys
import pdb
from golosio_recommendation_model.model.train.ann import prepare_posts, save_similar_posts
from golosio_recommendation_model.config import config
from annoy import AnnoyIndex
from sklearn.externals import joblib
from time import sleep

def get_posts(url, database):
  events = utils.get_events(url, database)
  utils.wait_and_lock_mutex(url, database, "inferred_vector")
  posts = utils.get_posts(url, database, events, {
    'inferred_vector' : {'$exists' : True},
    'similar_posts' : {'$exists' : False},
  })
  utils.unlock_mutex(url, database, "inferred_vector")
  return utils.preprocess_posts(posts, include_all_tags=True)

def estimate_number_of_features():
  return pd.read_csv(config['model_path'] + 'vectors.csv').shape[1] - 1

@utils.error_log("ANN predict")
def run_ann():
  database_url = config['database_url']
  database_name = config['database_name']
  utils.log("ANN predict", "Restore model...")
  utils.wait_for_file(config['model_path'] + 'popular_tags.pkl')
  popular_tags = joblib.load(config['model_path'] + "popular_tags.pkl")
  utils.wait_for_file(config['model_path'] + 'popular_categorical.pkl')
  popular_categorical = joblib.load(config['model_path'] + "popular_categorical.pkl")
  utils.wait_for_file(config['model_path'] + 'vectors.csv')
  model = AnnoyIndex(estimate_number_of_features()) 
  utils.wait_for_file(config['model_path'] + 'similar.ann')
  model.load(config['model_path'] + 'similar.ann')

  while True:
    utils.wait_between_iterations()
    utils.log("ANN predict", "Get posts...")
    posts = get_posts(database_url, database_name)
    if posts.shape[0] > 0:
      utils.log("ANN predict", "Prepare posts...")
      vectors, popular_tags, popular_categorical = prepare_posts(posts)
      utils.log("ANN predict", "Save similar posts...")
      save_similar_posts(database_url, database_name, posts, vectors, model)