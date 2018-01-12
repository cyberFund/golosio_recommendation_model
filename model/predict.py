import numpy as np
import pandas as pd
from pymongo import MongoClient, DESCENDING
import datetime as dt
from train import create_ffm_dataset, extend_events, get_posts
import ffm
from sklearn.externals import joblib
import utils
import sys
from tqdm import *
import pdb
import datetime as dt

USERS_POSTS_LIMIT = 100 # Max number of recommendations
HOURS_LIMIT = 60 * 24 # Time window for recommended posts

def get_new_posts(url, database):
  date = dt.datetime.now() - dt.timedelta(hours=HOURS_LIMIT)
  posts = pd.DataFrame()
  client = MongoClient(url)
  db = client[database]
  posts = pd.DataFrame(list(db.comment.find(
    {
      'permlink' : {'$exists' : True},
      'depth': 0,
      'topic': {'$exists' : True},
      'similar_posts': {'$exists' : True},
      'created': {'$gte': date}
    }, {
      'permlink': 1,
      'author': 1, 
      'topic' : 1,
      'topic_probability' : 1,
      'parent_permlink': 1,
      'created': 1,
      'json_metadata': 1,
      'similar_posts': 1,
      'similar_distances': 1,
    }
  )))
  return utils.preprocess_posts(posts)

def create_dataset(posts, events):
  posts = posts.set_index("post_permlink")
  dataset = pd.DataFrame(columns=["user_id", "post_permlink"])
  for user in tqdm(events["user_id"].unique()):
    user_events = events[(events["user_id"] == user) & (events["like"] > 0.5)] 
    similar_posts = [posts.loc[post]["similar_posts"] for post in user_events["post_permlink"] if post in posts.index]
    similar_posts = [post for posts in similar_posts for post in posts]
    similar_distances = [posts.loc[post]["similar_distances"] for post in user_events["post_permlink"] if post in posts.index]
    similar_distances = [distance for distances in similar_distances for distance in distances]
    seen_similar_posts = set(user_events["post_permlink"])
    unseen_similar_distances = np.array([float(distance) for index, distance in enumerate(similar_distances) if similar_posts[index] not in seen_similar_posts])
    unseen_similar_posts = [post for post in similar_posts if (post not in seen_similar_posts) and (post != "@/")]
    if len(unseen_similar_posts) > 0:
      unseen_similar_probabilities = np.array([1./len(unseen_similar_posts)] * len(unseen_similar_posts))
      if (unseen_similar_distances.sum() > 0):
        unseen_similar_probabilities = (unseen_similar_distances.max() - unseen_similar_distances) / ((unseen_similar_distances.max() - unseen_similar_distances).sum())
      selected_similar_posts = np.unique(np.random.choice(unseen_similar_posts, size=USERS_POSTS_LIMIT, p=unseen_similar_probabilities))
      user_dataset = pd.DataFrame()
      user_dataset["post_permlink"] = selected_similar_posts
      user_dataset["user_id"] = user
      dataset = pd.concat([dataset, user_dataset])
  dataset["like"] = 1
  return dataset

def save_recommendations(recommendations, url, database):
  recommendations.to_csv("recommendations.csv")
  posts = pd.DataFrame()
  client = MongoClient(url)
  db = client[database]
  db.recommendation.drop()
  db.recommendation.insert_many(recommendations.to_dict('records'))

def predict(events, database_url, database):
  print("Get new posts...")
  new_posts = get_new_posts(database_url, database)
  print("Get all posts...")
  posts = get_posts(database_url, database)
  print("Create dataset...")
  dataset = create_dataset(new_posts, events)
  print("Extend events...")
  dataset = extend_events(dataset, posts)
  print("Prepare model...")
  model = ffm.read_model("./model.bin")
  mappings = joblib.load("./mappings.pkl")
  mappings, ffm_dataset_X, ffm_dataset_y = create_ffm_dataset(dataset, mappings)
  ffm_dataset = ffm.FFMData(ffm_dataset_X, ffm_dataset_y)
  dataset["prediction"] = model.predict(ffm_dataset)
  print("Save recommendations...")
  save_recommendations(dataset[["user_id", "post_permlink", "prediction"]], database_url, database)

if (__name__ == "__main__"):
  raw_events = pd.read_csv("./extended_events.csv")
  predict(raw_events, sys.argv[1], sys.argv[2])
