import pandas as pd
from pymongo import MongoClient, DESCENDING
import datetime as dt
from train import create_ffm_dataset, extend_events
import ffm
from sklearn.externals import joblib
import utils
import sys
from tqdm import *
import pdb

USERS_POSTS_LIMIT = 100
MAX_POSTS_LIMIT = 100000

def get_new_posts(url, database):
  posts = pd.DataFrame()
  client = MongoClient(url)
  db = client[database]
  posts = pd.DataFrame(list(db.comment.find(
    {
      'permlink' : {'$exists' : True},
      'depth': 0,
      'topic': {'$exists' : True},
      'similar': {'$exists' : True}
    }, {
      'permlink': 1,
      'author': 1, 
      'topic' : 1,
      'topic_probability' : 1,
      'parent_permlink': 1,
      'created': 1,
      'json_metadata': 1,
      'similar': 1
    }
  ).sort("created", DESCENDING).limit(MAX_POSTS_LIMIT)))
  return utils.preprocess_posts(posts)

def create_dataset(posts, events):
  posts = posts.set_index("post_permlink")
  dataset = pd.DataFrame(columns=["user_id", "post_permlink"])
  for user in tqdm(events["user_id"].unique()):
    user_events = events[events["user_id"] == user] 
    pdb.set_trace()
    similar_posts = [posts.loc[post]["similar"] for post in user_events["post_permlink"] if post in posts.index]
    similar_posts = [post for posts in similar_posts for post in posts]
    if len(similar_posts) > 0:      
      selected_similar_posts = np.unique(np.random.choice(similar_posts, size=USERS_POSTS_LIMIT))
      user_dataset = pd.DataFrame()
      user_dataset["user_id"] = user
      user_dataset["post_permlink"] = selected_similar_posts
      dataset = pd.concat([dataset, user_dataset])
  dataset["like"] = 1
  return dataset

def save_recommendations(recommendations, url, database):
  posts = pd.DataFrame()
  client = MongoClient(url)
  db = client[database]
  db.recommendation.insert_many(recommendations.to_dict('records'))

def predict(events, database_url, database):
  print("Get new posts...")
  new_posts = get_new_posts(database_url, database)
  print("Create dataset...")
  dataset = create_dataset(new_posts, events)
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