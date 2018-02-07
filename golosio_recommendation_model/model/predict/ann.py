from golosio_recommendation_model.model import utils
import sys
import pdb
from golosio_recommendation_model.model.train.ann import prepare_posts
from golosio_recommendation_model.config import config
from annoy import AnnoyIndex
from sklearn.externals import joblib
from time import sleep

def get_posts(url, database):
  events = utils.get_events(url, database)
  posts = utils.get_posts(url, database, events, {
    'inferred_vector' : {'$exists' : True},
    'similar_posts' : {'$exists' : False},
  })
  return utils.preprocess_posts(posts, include_all_tags=True)

def estimate_number_of_features():
  return pd.read_csv(config['model_path'] + 'vectors.csv').shape[1] - 1

def save_similar_posts(url, database, posts, vectors, model):
  """
    Function to save similar posts for each post within created model to mongo database
  """
  client = MongoClient(url)
  db = client[database]
  for index in tqdm(posts.index):
    post = posts.loc[index]
    vector = vectors.loc[index].tolist()
    similar_indices, similar_distances = model.get_nns_by_vector(vector, NUMBER_OF_RECOMMENDATIONS, include_distances=True)
    similar_posts = posts.loc[similar_indices]["post_permlink"].tolist()
    db.comment.update_one({'_id': post["post_permlink"][1:]}, {'$set': {'similar_posts': similar_posts, 'committed_similar_posts': similar_posts, 'similar_distances': similar_distances, 'committed_similar_distances': similar_distances}})

@utils.error_log("ANN predict")
def predict_ann():
  database_url = config['database_url']
  database_name = config['database_name']
  utils.log("ANN predict", "Restore model...")
  popular_tags = joblib.load(config['model_path'] + "popular_tags.pkl")
  popular_categorical = joblib.load(config['model_path'] + "popular_categorical.pkl")
  model = AnnoyIndex(estimate_number_of_features()) 
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