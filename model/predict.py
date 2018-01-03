import pandas as pd
from pymongo import MongoClient, DESCENDING
import datetime as dt
from train import create_ffm_dataset, extend_events, prepare_raw_events
import ffm
from sklearn.externals import joblib
import utils
import sys
from tqdm import *
import pdb

POSTS_LIMIT = 10

def get_new_posts(url, database):
  posts = pd.DataFrame()
  client = MongoClient(url)
  db = client[database]
  posts = pd.DataFrame(list(db.comment.find(
    {
      'permlink' : {'$exists' : True},
      'depth': 0,
      'topic': {'$exists' : True},
    }, {
      'permlink': 1,
      'author': 1, 
      'topic' : 1,
      'topic_probability' : 1,
      'parent_permlink': 1,
      'created': 1,
      'json_metadata': 1
    }
  ).sort("created", DESCENDING).limit(POSTS_LIMIT)))
  return utils.preprocess_posts(posts)

def create_dataset(posts, users):
  posts_for_join = pd.DataFrame()
  posts_for_join["post_permlink"] = posts["post_permlink"].unique()
  posts_for_join["like"] = 1
  users_for_join = pd.DataFrame()
  users_for_join["user_id"] = users
  users_for_join["like"] = 1
  dataset = pd.merge(posts_for_join, users_for_join, on='like')
  return dataset

def save_recommendations(recommendations, url, database):
  posts = pd.DataFrame()
  client = MongoClient(url)
  db = client[database]
  db.recommendation.insert_many(recommendations.to_dict('records'))

def predict(events, database_url, database):
  events = prepare_raw_events(events)
  users = events["user_id"].unique()
  print("Get new posts...")
  new_posts = get_new_posts(database_url, database)
  print("Create dataset...")
  dataset = create_dataset(new_posts, users)
  print("Extend events...")
  dataset = extend_events(dataset, new_posts)
  print("Prepare model...")
  model = ffm.read_model("./model.bin")
  mappings = joblib.load("./mappings.pkl")
  mappings, ffm_dataset_X, ffm_dataset_y = create_ffm_dataset(dataset, mappings)
  ffm_dataset = ffm.FFMData(ffm_dataset_X, ffm_dataset_y)
  dataset["prediction"] = model.predict(ffm_dataset)
  print("Save recommendations...")
  save_recommendations(dataset[["user_id", "post_permlink", "prediction"]], database_url, database)

if (__name__ == "__main__"):
  raw_events = pd.read_csv(sys.argv[1])
  predict(raw_events, sys.argv[2], sys.argv[3])