import pdb
import numpy as np
import utils
import sys
from pymongo import MongoClient
import pandas as pd
import pdb
from tqdm import *
from annoy import AnnoyIndex

NUMBER_OF_TREES = 1000

def get_posts(url, database):
  client = MongoClient(url)
  db = client[database]
  posts = pd.DataFrame(list(db.comment.find(
    {
      'permlink' : {'$exists' : True},
      'topic' : {'$exists' : True},
      'depth': 0,
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
  return utils.preprocess_posts(posts, include_all_tags=True)

def add_popular_tags(posts):
  all_tags, tag_counts = np.unique([tag for tags in posts["tags"] for tag in tags], return_counts=True)
  popular_tags = all_tags[np.argsort(-tag_counts)][0:100]
  for tag in tqdm(popular_tags):
    posts[tag + "_tag"] = posts["tags"].apply(lambda x: tag in x)
  posts["another_tags"] = posts["tags"].apply(lambda x: len([t for t in x if t not in popular_tags]) > 0)
  return posts.drop(["tags"], axis=1)

def convert_categorical(posts):
  categorical_columns = ["author", "parent_permlink"]
  for column in categorical_columns:
    posts[column] = posts[column].astype("category").cat.codes
  return posts

def convert_numerical(posts):
  float_columns = ["probability", "coefficient"]
  for column in posts.columns:
      if column not in float_columns:
          posts[column] = posts[column].astype(int)
  return posts

def prepare_posts(posts):
  posts = posts.drop(['body', 'created', 'permlink', 'post_permlink'], axis=1)
  posts = add_popular_tags(posts)
  posts = convert_categorical(posts)
  posts = convert_numerical(posts)
  return posts

def train_model(model):
  model.build(1000)

def create_model(posts):
  factors = posts.shape[1]
  trees = AnnoyIndex(factors) 
  for index, row in posts.iterrows():
    trees.add_item(index, row.tolist())
  return trees
  
def save_similar_posts(url, database, posts, vectors, model):
  client = MongoClient(url)
  db = client[database]
  for index in tqdm(posts.index):
    post = posts.loc[index]
    similar_indices, similar_distances = model.get_nns_by_vector(vectors.loc[index].tolist(), 5, include_distances=True)
    similar_posts = posts.loc[similar_indices]["post_permlink"].tolist()
    db.comment.update_one({'_id': post["post_permlink"][1:]}, {'$set': {'similar_posts': similar_posts, 'similar_distances': similar_distances}})

def run_ann(database_url, database_name):
  print("Get posts...")
  posts = get_posts(database_url, database_name)
  print("Prepare posts...")
  vectors = prepare_posts(posts)
  print("Prepare model...")
  model = create_model(vectors)
  print("Train model...")
  train_model(model)
  model.save("similar.ann")
  print("Save similar posts...")
  save_similar_posts(database_url, database_name, posts, vectors, model)

if (__name__ == "__main__"):
  run_ann(sys.argv[1], sys.argv[2])