import pandas as pd
from pymongo import MongoClient
import datetime as dt
from model.train import create_ffm_dataset, extend_events
import ffm
from sklearn.externals import joblib

def get_new_posts(url, database):
  date = dt.datetime.now() - dt.timedelta(hours=5)
  posts = pd.DataFrame()
  client = MongoClient(url)
  db = client[database]
  posts = pd.DataFrame(list(db.comment.find(
    {
      'permlink' : {'$exists' : True},
      'depth': 0,
      'created': {'$gte': date},
    }, {
      'permlink': 1,
      'author': 1, 
      'topic' : 1,
      'topic_probability' : 1,
      'parent_permlink': 1,
      'created': 1,
      'json_metadata': 1
    }
  )))
  posts["post_permlink"] = "@" + posts["author"] + "/" + posts["permlink"]
  posts["first_tag"] = posts["json_metadata"].apply(lambda x: x["tags"][0])
  posts["last_tag"] = posts["json_metadata"].apply(lambda x: x["tags"][-1])
  return posts.drop(["permlink", "json_metadata", "_id"], axis=1)

def create_dataset(posts, users):
  dataset = pd.DataFrame(columns=list(posts.columns))
  for user in users:
    posts["user_id"] = user
    dataset = pd.concat([dataset, posts], axis=0)
  dataset["like"] = 1
  extend_events(dataset, posts)
  return dataset

def save_recommendations(recommendations, url, database):
  posts = pd.DataFrame()
  client = MongoClient(url)
  db = client[database]
  db.recommendation.insert_many(recommendations.to_dict('records'))

def predict(events, database_url, database):
  users = events["user_id"].unique()
  new_posts = get_new_posts(database_url, database)
  dataset = create_dataset(new_posts, users)
  model = ffm.read_model("./model.bin")
  mappings = joblib.load("./mappings.pkl")
  mappings, ffm_dataset_X, ffm_dataset_y = create_ffm_dataset(dataset, mappings)
  ffm_dataset = ffm.FFMData(ffm_dataset_X, ffm_dataset_y)
  dataset["prediction"] = model.predict(ffm_dataset)
  save_recommendations(dataset[["user_id", "post_permlink", "prediction"]], database_url, database)