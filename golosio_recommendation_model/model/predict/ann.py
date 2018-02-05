from golosio_recommendation_model.model import utils
import sys
import pdb
from golosio_recommendation_model.model.train.ann import prepare_posts, save_similar_posts
from annoy import AnnoyIndex

def get_posts(url, database):
  events = utils.get_events(url, database)
  posts = utils.get_posts(url, database, events, {
    'inferred_vector' : {'$exists' : True},
    'similar_posts' : {'$exists' : False},
  })
  return utils.preprocess_posts(posts, include_all_tags=True)

@utils.error_log("ANN predict")
def run_ann(database_url, database_name):
  utils.log("ANN predict", "Get posts...")
  posts = get_posts(database_url, database_name)
  if posts.shape[0] > 0:
    utils.log("ANN predict", "Prepare posts...")
    vectors = prepare_posts(posts)
    utils.log("ANN predict", "Restore model...")
    model = AnnoyIndex(vectors.shape[1])
    model.load('similar.ann')
    utils.log("ANN predict", "Save similar posts...")
    save_similar_posts(database_url, database_name, posts, vectors, model)

if (__name__ == "__main__"):
  run_ann(sys.argv[1], sys.argv[2])