import pdb
import numpy as np
import utils
import sys
from pymongo import MongoClient
import pandas as pd
import pdb
from tqdm import *
from annoy import AnnoyIndex
from sklearn.preprocessing import quantile_transform
import datetime as dt

NUMBER_OF_TREES = 1000
NUMBER_OF_RECOMMENDATIONS = 10
NUMBER_OF_VALUES = 1000

HOURS_LIMIT = 14 * 24 # Time window for analyzed posts

def get_posts(url, database):
  """
    Function to get last posts with defined inferred vector and topic from mongo database
  """
  date = utils.get_last_post_date(url, database) - dt.timedelta(hours=HOURS_LIMIT)
  client = MongoClient(url)
  db = client[database]
  posts = pd.DataFrame(list(db.comment.find(
    {
      'permlink' : {'$exists' : True},
      'inferred_vector' : {'$exists' : True},
      'depth': 0,
      'created': {'$gte': date}
    }, {
      'permlink': 1,
      'author': 1,
      'inferred_vector': 1,
      'parent_permlink': 1,
      'created': 1,
      'json_metadata': 1,
      'body': 1,
    }
  )))
  return utils.preprocess_posts(posts, include_all_tags=True)

def add_popular_tags(posts):
  """
    Function to encode tags with only popular values
  """
  all_tags, tag_counts = np.unique([tag for tags in posts["tags"] for tag in tags], return_counts=True)
  popular_tags = all_tags[np.argsort(-tag_counts)][0:NUMBER_OF_VALUES]
  for tag in tqdm(popular_tags):
    posts[tag + "_tag"] = posts["tags"].apply(lambda x: tag in x).astype(int)
  posts["another_tags"] = posts["tags"].apply(lambda x: len([t for t in x if t not in popular_tags]) > 0).astype(int)
  return posts.drop(["tags"], axis=1)

def convert_categorical(posts):
  """
    Function to one-hot encode categorical columns with only popular values
  """
  categorical_columns = ["author", "parent_permlink"]
  for column in categorical_columns:
    all_values, value_counts = np.unique(posts[column].tolist(), return_counts=True)
    popular_values = all_values[np.argsort(-value_counts)][0:NUMBER_OF_VALUES]
    for value in tqdm(popular_values):
        posts[str(value) + "_" + column] = posts[column].apply(lambda x: x == value).astype(int)
    posts["another_" + column] = posts[column].apply(lambda x: x not in popular_values).astype(int)
    posts = posts.drop([column], axis=1)
  return posts

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

def prepare_posts(posts):
  """
    Function to vectorise posts for ANN algorithm
  """
  posts = posts.drop(['body', 'permlink', 'post_permlink', 'created'], axis=1)
  posts = add_popular_tags(posts)
  posts = convert_categorical(posts)
  posts = convert_array(posts)
  return posts

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
  for index in tqdm(posts.index):
    post = posts.loc[index]
    similar_indices, similar_distances = model.get_nns_by_item(index, NUMBER_OF_RECOMMENDATIONS, include_distances=True)
    similar_posts = posts.loc[similar_indices]["post_permlink"].tolist()
    db.comment.update_one({'_id': post["post_permlink"][1:]}, {'$set': {'similar_posts': similar_posts, 'similar_distances': similar_distances}})

@utils.error_log("ANN")
def run_ann(database_url, database_name):
  """
    Function to run ANN process:
    - Get posts from mongo
    - Vectorize posts
    - Create and train ANN model from vectorized posts
    - Save similar posts to mongo database
  """
  utils.wait_and_lock_mutex(database_url, database_name, "doc2vec")
  utils.wait_and_lock_mutex(database_url, database_name, "lda")
  utils.log("ANN", "Get posts...")
  posts = get_posts(database_url, database_name)
  utils.unlock_mutex(database_url, database_name, "doc2vec")
  utils.unlock_mutex(database_url, database_name, "lda")
  utils.log("ANN", "Prepare posts...")
  vectors = prepare_posts(posts)
  vectors.to_csv("./vectors.csv")
  utils.log("ANN", "Prepare model...")
  model = create_model(vectors)
  utils.log("ANN", "Train model...")
  train_model(model)
  model.save("similar.ann")
  utils.wait_and_lock_mutex(database_url, database_name, "ann")
  utils.log("ANN", "Save similar posts...")
  save_similar_posts(database_url, database_name, posts, vectors, model)
  utils.unlock_mutex(database_url, database_name, "ann")

if (__name__ == "__main__"):
  run_ann(sys.argv[1], sys.argv[2])
