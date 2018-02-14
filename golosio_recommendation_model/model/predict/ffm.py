import numpy as np
import pandas as pd
from pymongo import MongoClient, DESCENDING
import datetime as dt
from golosio_recommendation_model.model.train.ffm import create_ffm_dataset, extend_events
import ffm
from sklearn.externals import joblib
from golosio_recommendation_model.model import utils
import sys
from tqdm import *
import pdb
import datetime as dt
from golosio_recommendation_model.config import config

USERS_POSTS_LIMIT = 100 # Max number of recommendations

def get_posts(url, database):
  events = utils.get_events(url, database)
  posts = utils.get_posts(url, database, events, {
    'similar_posts' : {'$exists' : True}
  })
  return utils.preprocess_posts(posts)

def create_dataset(posts, events):
  """
    Function to generate pairs with posts similar to viewed for each user
  """
  posts = posts.set_index("post_permlink")
  dataset = pd.DataFrame(columns=["user_id", "post_permlink"])
  for user in tqdm(events["user_id"].unique()):
    user_events = events[(events["user_id"] == user) & (events["like"] > 0.5)]  # Use only views and comments
    seen_posts = set(user_events["post_permlink"])
    unseen_posts = [post for post in set(posts.index) if (post not in seen_posts) and (post != "@/")]
    user_dataset = pd.DataFrame()
    user_dataset["post_permlink"] = unseen_posts
    user_dataset["user_id"] = user
    dataset = pd.concat([dataset, user_dataset])
  dataset["like"] = 1
  return dataset

def save_recommendations(recommendations, url, database):
  recommendations.to_csv(config['model_path'] + "recommendations.csv")
  client = MongoClient(url)
  db = client[database]
  for index, recommendation in recommendations.iterrows():
    db.recommendation.update_one({'user_id': recommendation["user_id"], 'post_permlink': recommendation["post_permlink"]}, {'$set': {'prediction': recommendation['prediction']}}, upsert=True)

@utils.error_log("FFM predict")
def predict_ffm():
  """
    Function to run prediction process:
    - Get all posts in a model
    - Get only new posts
    - Generate pairs with similar posts for each user
    - Load a model from disk
    - Get FFM predictions
    - Save recommendations to a mongo database
  """
  database_url = config['database_url']
  database = config['database_name']
  utils.log("FFM predict", "Prepare model...")
  model = ffm.read_model(config['model_path'] + "model.bin")
  mappings = joblib.load(config['model_path'] + "mappings.pkl")
  
  while True:
    utils.log("FFM predict", "Get posts...")
    posts = get_posts(database_url, database)
    utils.log("FFM predict", "Create dataset...")
    events = utils.get_events(database_url, database)
    dataset = create_dataset(posts, events)
    utils.log("FFM predict", "Extend events...")
    dataset = extend_events(dataset, posts)
    mappings, ffm_dataset_X, ffm_dataset_y = create_ffm_dataset(dataset, mappings)
    ffm_dataset = ffm.FFMData(ffm_dataset_X, ffm_dataset_y)
    dataset["prediction"] = model.predict(ffm_dataset)
    utils.log("FFM predict", "Save recommendations...")
    save_recommendations(dataset[["user_id", "post_permlink", "prediction"]], database_url, database)
