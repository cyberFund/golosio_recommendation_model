import pdb
import numpy as np
from golosio_recommendation_model.model import utils
import sys
from pymongo import MongoClient
import pandas as pd
import pdb
from tqdm import *
from annoy import AnnoyIndex
from sklearn.preprocessing import quantile_transform
import datetime as dt
from sklearn.externals import joblib
from golosio_recommendation_model.config import config

NUMBER_OF_TREES = 1000
NUMBER_OF_RECOMMENDATIONS = 10
NUMBER_OF_VALUES = 1000

def get_posts(url, database):
  events = utils.get_events(url, database)
  utils.wait_and_lock_mutex(url, database, "inferred_vector")
  posts = utils.get_posts(url, database, events, {
    'inferred_vector' : {'$exists' : True}
  })
  utils.unlock_mutex(url, database, "inferred_vector")
  return utils.preprocess_posts(posts, include_all_tags=True)

def add_popular_tags(posts, popular_tags):
  """
    Function to encode tags with only popular values
  """
  if popular_tags is None:
    all_tags, tag_counts = np.unique([tag for tags in posts["tags"] for tag in tags], return_counts=True)
    popular_tags = all_tags[np.argsort(-tag_counts)][0:NUMBER_OF_VALUES]
  for tag in tqdm(popular_tags):
    posts[tag + "_tag"] = posts["tags"].apply(lambda x: tag in x).astype(int)
  posts["another_tags"] = posts["tags"].apply(lambda x: len([t for t in x if t not in popular_tags]) > 0).astype(int)
  return posts.drop(["tags"], axis=1), popular_tags

def convert_categorical(posts, popular_categorical):
  """
    Function to one-hot encode categorical columns with only popular values
  """
  categorical_columns = ["author", "parent_permlink"]
  for column in categorical_columns:
    if column not in popular_categorical.keys():
      all_values, value_counts = np.unique(posts[column].tolist(), return_counts=True)
      popular_values = all_values[np.argsort(-value_counts)][0:NUMBER_OF_VALUES]
    else:
      popular_values = popular_categorical[column]
    popular_categorical[column] = popular_values
    for value in tqdm(popular_values):
        posts[str(value) + "_" + column] = posts[column].apply(lambda x: x == value).astype(int)
    posts["another_" + column] = posts[column].apply(lambda x: x not in popular_values).astype(int)
    posts = posts.drop([column], axis=1)
  return posts, popular_categorical

def convert_array(posts):
  """
    Function to convert columns with float arrays to a set of columns
  """
  array_columns = ["inferred_vector"]
  for column in array_columns:
    length = len(posts[column].iloc[0])
    values = [str(x) + "_" + column for x in range(length)]
    posts[values] = pd.DataFrame(posts[column].values.tolist(), index=posts.index)
    posts = posts.drop([column], axis=1)
  return posts

def convert_dates(posts):
  """
    Function to convert dates to coefficients from 0 to 1
  """
  dates_columns = ["created"]
  for column in dates_columns:
    posts[column] = pd.to_datetime(posts[column]).apply(lambda x: x.value)
    posts[column] = quantile_transform(posts[column].values.reshape(-1, 1)).reshape(-1)
  return posts

def prepare_posts(posts, popular_tags=None, popular_categorical={}):
  """
    Function to vectorise posts for ANN algorithm
  """
  posts = posts.drop(['body', 'permlink', 'post_permlink', 'created', 'similar_posts', 'similar_distances', 'prepared_body'], axis=1)
  posts, popular_tags = add_popular_tags(posts, popular_tags)
  posts, popular_categorical = convert_categorical(posts, popular_categorical)
  posts = convert_array(posts)
  return posts, popular_tags, popular_categorical

def train_model(model):
  """
    Function to train ANN model
  """
  model.build(NUMBER_OF_TREES)

def create_model(posts):
  """
    Function to create ANN model from vectors
  """
  factors = posts.shape[1]
  trees = AnnoyIndex(factors) 
  for index, row in posts.iterrows():
    trees.add_item(index, row.tolist())
  return trees

def save_similar_posts(url, database, posts, vectors, model):
  """
    Function to save similar posts for each post within created model to mongo database
  """
  client = MongoClient(url)
  db = client[database]
  utils.wait_and_lock_mutex(url, database, "similar_posts")
  for index in tqdm(posts.index):
    post = posts.loc[index]
    vector = vectors.loc[index].tolist()
    similar_indices, similar_distances = model.get_nns_by_vector(vector, NUMBER_OF_RECOMMENDATIONS, include_distances=True)
    similar_posts = posts.loc[similar_indices]["post_permlink"].tolist()
    db.comment.update_one({'_id': post["post_permlink"][1:]}, {'$set': {'similar_posts': similar_posts, 'similar_distances': similar_distances}})
  utils.unlock_mutex(url, database, "similar_posts")

@utils.error_log("ANN train")
def run_ann(database_url, database_name):
  """
    Function to run ANN process:
    - Get posts from mongo
    - Vectorize posts
    - Create and train ANN model from vectorized posts
    - Save similar posts to mongo database
  """
  utils.log("ANN train", "Get posts...")
  posts = get_posts(database_url, database_name)
  utils.log("ANN train", "Prepare posts...")
  vectors, popular_tags, popular_categorical = prepare_posts(posts)
  joblib.dump(popular_tags, config['model_path'] + "popular_tags.pkl")
  joblib.dump(popular_categorical, config['model_path'] + "popular_categorical.pkl")
  vectors.to_csv(config['model_path'] + "vectors.csv")
  utils.log("ANN train", "Prepare model...")
  model = create_model(vectors)
  utils.log("ANN train", "Train model...")
  train_model(model)
  model.save(config['model_path'] + "similar.ann")
  utils.log("ANN train", "Save similar posts...")
  all_posts = get_posts(database_url, database_name)
  all_vectors, popular_tags, popular_categorical = prepare_posts(all_posts, popular_tags, popular_categorical)
  save_similar_posts(database_url, database_name, all_posts, all_vectors, model)

if (__name__ == "__main__"):
  run_ann(sys.argv[1], sys.argv[2])
