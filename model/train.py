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
import utils
import sys
import dask.dataframe as dd
from tqdm import *
import datetime as dt

MODEL_PARAMETERS = {
  'eta': 0.1, 
  'lam': 0.01, 
  'k': 70
}

ITERATIONS = 10
WORKERS = 13
HOURS_LIMIT = 14 * 24 # Time window for analyzed posts

def parse_refurl(url):
  """
    Function to fix refurls in events
  """
  return "/".join(url.split("/")[4:])

def parse_recommendations(urls):
  """
    Function to parse arrays with recommendations for each event
  """
  return ["@" + url[1:-1] for url in urls[1:-1].split(",") if len(url) > 0]

def get_raw_events(url, database): 
  """
    Function to get latest events from a database
  """
  date = utils.get_last_event_date(url, database) - dt.timedelta(hours=HOURS_LIMIT)
  client = MongoClient(url) 
  db = client[database] 
  events = pd.DataFrame(list(db.event.find( 
    {
      'created_at': {'$gte': date}
    }, { 
      'event_type': 1,  
      'value' : 1, 
      'user_id' : 1, 
      'refurl': 1, 
      'status': 1, 
      'created_at': 1 
    } 
  ))) 
  return events 

def prepare_raw_events(raw_events):
  """
    Function to fix some problems with events after mysql export
  """
  raw_events["refurl"] = raw_events["refurl"].astype(str)
  raw_events["value"] = raw_events["value"].astype(str)
  raw_events["user_id"].fillna("\\N", inplace=True)
  return raw_events[raw_events["user_id"].astype(str) != "\\N"]

def get_user_events(raw_events):
  """
    Function to generate csv file with recommendations, views, votes and comments lists for each user
  """
  user_events = pd.DataFrame(columns=["user", "recommendations", "views", "votes", "comments"])
  users = raw_events["user_id"].unique()
  users_raw_events = raw_events.groupby("user_id")
  recommendations = []
  views = []
  votes = []
  comments = []
  for user in tqdm(users):
    user_raw_events = users_raw_events.get_group(user)
    user_recommendations = [parse_recommendations(x) for x in user_raw_events[user_raw_events["event_type"] == "PageView"]["value"]]
    recommendations.append(set(item for sublist in user_recommendations for item in sublist))
    views.append(set(parse_refurl(x) for x in user_raw_events["refurl"] if x.count("/") >= 5))
    votes.append(set(x for x in user_raw_events[(user_raw_events["event_type"] == "Vote")]["value"]))
    comments.append(set(parse_refurl(x) for x in user_raw_events[(user_raw_events["event_type"] == "Comment")]["refurl"]))
  user_events["user"] = users
  user_events["views"] = views
  user_events["votes"] = votes
  user_events["comments"] = comments
  user_events["recommendations"] = recommendations
  return user_events

def get_posts(url, database):
  """
    Function to get all posts from a database
  """
  client = MongoClient(url)
  db = client[database]
  posts = pd.DataFrame(list(db.comment.find(
    {
      'permlink' : {'$exists' : True},
      'depth': 0,
    }, {
      'permlink': 1,
      'author': 1, 
      'parent_permlink': 1,
      'created': 1,
      'json_metadata': 1
    }
  )))
  return utils.preprocess_posts(posts)

def get_coefficient(user_events, user, post):
  """
    Function to find a coefficient for each user-post pair
      - Comment - 1
      - View without comment - 0.7
      - Recommendation without view - -1
      - Other - 0
  """
  if user not in user_events.index:
    return 0
  user_event = user_events.loc[user]
  if (post in user_event["comments"]):
    return 1
  elif ((post in user_event["views"])):
    return 0.7
  elif (post in user_event["recommendations"]):
    return -1
  else:
    return 0

def get_events(user_events):
  """
    Function to generate user-post pairs with a coefficient depending on user sympathy to this post
  """
  events = pd.DataFrame()
  users = []
  posts = []
  likes = []
  user_events = user_events.set_index("user")
  for user in tqdm(user_events.index):
    user_event = user_events.loc[user]
    event_posts = list(user_event["views"]) + list(user_event["votes"]) + list(user_event["comments"]) + list(user_event["recommendations"])
    for post in set(event_posts):
      if post != "":
        users.append(user)
        posts.append(post)
  events["user_id"] = users
  events["post_permlink"] = posts
  distributed_events = dd.from_pandas(events, npartitions=WORKERS)
  events["like"] = distributed_events.apply(lambda x: get_coefficient(user_events, x["user_id"], x["post_permlink"]), axis=1).compute()
  return events

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
  return model, roc_auc_score(train_y, model.predict(train_ffm_data)), roc_auc_score(test_y, model.predict(test_ffm_data))

@utils.error_log("FFM")
def train(raw_events, database_url, database):
  """
    Function to train FFM model
    - Get all events from mongo database
    - Convert them to a set of unique user-post pairs with a coefficient depending on user sympathy
    - Extend events with a posts info
    - Convert events to a format for FFM algorithm
    - Build model with chosen train and test set
    - Save trained model
  """
  utils.log("FFM", "Prepare raw events...")
  raw_events = prepare_raw_events(raw_events)
  utils.log("FFM", "Prepare user events...")
  user_events = get_user_events(raw_events)

  user_events.to_csv("user_events.csv")
  # user_events = pd.read_csv("user_events.csv")
  # user_events.recommendations = user_events.recommendations.apply(lambda x: eval(x))
  # user_events.views = user_events.views.apply(lambda x: eval(x))
  # user_events.votes = user_events.votes.apply(lambda x: eval(x))
  # user_events.comments = user_events.comments.apply(lambda x: eval(x))

  utils.log("FFM", "Prepare events...")
  events = get_events(user_events)

  events.to_csv("prepared_events.csv")
  # events = pd.read_csv("prepared_events.csv").drop(["Unnamed: 0"], axis=1)

  utils.wait_and_lock_mutex(database_url, database, "lda")
  utils.log("FFM", "Prepare posts...")
  posts = get_posts(database_url, database)
  utils.unlock_mutex(database_url, database, "lda")

  posts.to_csv("prepared_posts.csv")
  # posts = pd.read_csv("prepared_posts.csv").drop(["Unnamed: 0"], axis=1)

  utils.log("FFM", "Extend events...")
  events = extend_events(events, posts)

  utils.log("FFM", "Save events...")
  events.to_csv("extended_events.csv")

  # events = pd.read_csv("extended_events.csv").drop(["Unnamed: 0"], axis=1)

  utils.log("FFM", "Create ffm dataset...")
  mappings, X, y = create_ffm_dataset(events)
  joblib.dump(X, "./X.pkl")
  joblib.dump(y, "./y.pkl")
  train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
  utils.log("FFM", "Build model...")
  model, train_auc_roc, test_auc_roc = build_model(train_X, train_y, test_X, test_y)
  utils.log("FFM", train_auc_roc)
  utils.log("FFM", test_auc_roc)
  model.save_model("./model.bin")
  joblib.dump(mappings, "./mappings.pkl")

if (__name__ == "__main__"):
  raw_events = get_raw_events(sys.argv[1], sys.argv[2])
  # raw_events = raw_events.sample(int(raw_events.shape[0]/1000))
  train(raw_events, sys.argv[1], sys.argv[2])
