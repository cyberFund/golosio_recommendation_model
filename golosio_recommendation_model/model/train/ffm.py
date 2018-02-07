import numpy as np
import ffm
from sklearn.metrics import roc_auc_score
from pymongo import MongoClient
import pandas as pd
import pdb
from sklearn.externals import joblib
from sklearn.preprocessing import quantile_transform
from sklearn.model_selection import train_test_split
import numpy as np
from golosio_recommendation_model.model import utils
import sys
import dask.dataframe as dd
from tqdm import *
import datetime as dt
from golosio_recommendation_model.config import config

MODEL_PARAMETERS = {
  'eta': 0.1, 
  'lam': 0.01, 
  'k': 70
}

ITERATIONS = 10
WORKERS = 13

def get_posts(url, database, events):
  posts = utils.get_posts(url, database, events)
  return utils.preprocess_posts(posts)

def extend_events(events, posts):
  """
    Function to extend each event with a post columns
    Uses quantile transform for numeric features like post popularity or event time
  """
  posts = posts.set_index("post_permlink")
  posts["created"] = pd.to_datetime(posts["created"])
  events = events.set_index("post_permlink")
  popularity = events.groupby("post_permlink").count()
  popularity["popularity"] = popularity["like"]
  events = events.join(posts).join(popularity[["popularity"]]).reset_index()
  events["popularity_coefficient"] = quantile_transform(events["popularity"].values.reshape(-1, 1), output_distribution="normal", copy=True).reshape(-1)
  return events

def create_mapping(series):
  """
    Function to create integer mapping for each unique value in a given series
  """
  series = series.fillna("")
  mapping = {}
  for (idx, mid) in enumerate(np.unique(series)):
    mapping[mid] = idx
  return mapping

def create_ffm_row(mapping, event):
  """
    Function to create row of dataset in a format for FFM model (See here https://github.com/alexeygrigorev/libffm-python) 
  """
  return [
    (0, mapping["uid_to_idx"].get(event["user_id"], max(mapping["uid_to_idx"].values()) + 1), 1),
    (1, mapping["pid_to_idx"].get(event["post_permlink"], max(mapping["pid_to_idx"].values()) + 1), 1),
    (2, mapping["aid_to_idx"].get(event["author"], max(mapping["aid_to_idx"].values()) + 1), 1),
    (3, mapping["parid_to_idx"].get(event["parent_permlink"], max(mapping["parid_to_idx"].values()) + 1), 1),
    (4, mapping["ftgid_to_idx"].get(event["first_tag"], max(mapping["ftgid_to_idx"].values()) + 1), 1),
    (5, mapping["ltgid_to_idx"].get(event["last_tag"], max(mapping["ltgid_to_idx"].values()) + 1), 1),
    (6, 1, event["popularity_coefficient"]),
  ]

def create_ffm_dataset(events, mapping=None):
  """
    Function to create dataset for each event
    Comments and views are positive samples, everything else are negatives
  """
  if not mapping:
    mapping = {}
    mapping["uid_to_idx"] = create_mapping(events["user_id"])
    mapping["pid_to_idx"] = create_mapping(events["post_permlink"])
    mapping["aid_to_idx"] = create_mapping(events["author"])
    mapping["parid_to_idx"] = create_mapping(events["parent_permlink"])
    mapping["ftgid_to_idx"] = create_mapping(events["first_tag"])
    mapping["ltgid_to_idx"] = create_mapping(events["last_tag"])

  # TODO get rid of this hack (problem with interpreting list of tuples in .apply function for a whole dataframe)
  events["index"] = range(events.shape[0])
  distributed_events = dd.from_pandas(events, npartitions=WORKERS)
  events = events.set_index("index")
  result = distributed_events["index"].apply(lambda x: create_ffm_row(mapping, events.loc[x])).compute()
  return mapping, result, (events["like"] >= 0.7).tolist()

def build_model(train_X, train_y, test_X, test_y):
  """
    Function to build and to train model from given train and test dataset
  """
  train_ffm_data = ffm.FFMData(train_X, train_y)
  test_ffm_data = ffm.FFMData(test_X, test_y)

  model = ffm.FFM(**MODEL_PARAMETERS)
  model.init_model(train_ffm_data)

  for i in range(ITERATIONS):
    model.iteration(train_ffm_data)
  # TODO temporary fix. replace this line of code with a commented line
  # return model, roc_auc_score(train_y, model.predict(train_ffm_data)), roc_auc_score(test_y, model.predict(test_ffm_data))
  return model, 1, 1

@utils.error_log("FFM train")
def run_ffm(database_url, database):
  """
    Function to train FFM model
    - Get all events from mongo database
    - Convert them to a set of unique user-post pairs with a coefficient depending on user sympathy
    - Extend events with a posts info
    - Convert events to a format for FFM algorithm
    - Build model with chosen train and test set
    - Save trained model
  """
  utils.log("FFM train", "Prepare events...")
  events = utils.get_events(database_url, database)

  events.to_csv(config['model_path'] + "prepared_events.csv")
  # events = pd.read_csv("prepared_events.csv").drop(["Unnamed: 0"], axis=1)

  utils.log("FFM train", "Prepare posts...")
  posts = get_posts(database_url, database, events)

  posts.to_csv(config['model_path'] + "prepared_posts.csv")
  # posts = pd.read_csv("prepared_posts.csv").drop(["Unnamed: 0"], axis=1)

  utils.log("FFM train", "Extend events...")
  events = extend_events(events, posts)

  utils.log("FFM train", "Save events...")
  events.to_csv(config['model_path'] + "extended_events.csv")

  # events = pd.read_csv("extended_events.csv").drop(["Unnamed: 0"], axis=1)

  utils.log("FFM train", "Create ffm dataset...")
  mappings, X, y = create_ffm_dataset(events)
  joblib.dump(X, config['model_path'] + "X.pkl")
  joblib.dump(y, config['model_path'] + "y.pkl")
  train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
  utils.log("FFM train", "Build model...")
  model, train_auc_roc, test_auc_roc = build_model(train_X, train_y, test_X, test_y)
  utils.log("FFM train", train_auc_roc)
  utils.log("FFM train", test_auc_roc)
  model.save_model(config['model_path'] + "model.bin")
  joblib.dump(mappings, config['model_path'] + "mappings.pkl")

if (__name__ == "__main__"):
  train(sys.argv[1], sys.argv[2])
