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

MODEL_PARAMETERS = {
  'eta': 0.1, 
  'lam': 0.01, 
  'k': 70
}

ITERATIONS = 10

def parse_refurl(url):
  return "/".join(url.split("/")[4:])

def parse_recommendations(urls):
  return ["@" + url[1:-1] for url in urls[1:-1].split(",") if len(url) > 0]

def get_user_events(raw_events):
  user_events = pd.DataFrame(columns=["user", "recommendations", "views", "votes", "comments"])
  users = raw_events["user_id"].unique()
  raw_events["refurl"] = raw_events["refurl"].astype(str)
  raw_events["value"] = raw_events["value"].astype(str)
  recommendations = []
  views = []
  votes = []
  comments = []
  for user in users:
    user_raw_events = raw_events[raw_events["user_id"] == user]
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
  )))
  return utils.preprocess_posts(posts)

def get_events(user_events):
  events = pd.DataFrame(columns=["user_id"])
  users = []
  posts = []
  likes = []
  user_events = user_events.set_index("user")
  for user in user_events.index:
    user_event = user_events.loc[user]
    event_posts = list(user_event["views"]) + list(user_event["votes"]) + list(user_event["comments"]) + list(user_event["recommendations"])
    for post in set(event_posts):
      users.append(user)
      posts.append(post)
      if (post in user_event["comments"]):
        likes.append(1)
      elif ((post in user_event["views"])):
        likes.append(0.7)
      elif (post in user_event["recommendations"]):
        likes.append(-1)
      else:
        likes.append(0)
  events["user_id"] = users
  events["post_permlink"] = posts
  events["like"] = likes
  return events

def extend_events(events, posts):
  posts = posts.set_index("post_permlink")
  posts["created"] = pd.to_datetime(posts["created"])
  events["parent_permlink"] = events["post_permlink"].apply(lambda x: posts.loc[x]["parent_permlink"] if x in posts.index else "")
  events["author"] = events["post_permlink"].apply(lambda x: posts.loc[x]["author"] if x in posts.index else "")
  events["first_tag"] = events["post_permlink"].apply(lambda x: posts.loc[x]["first_tag"] if x in posts.index else "")
  events["last_tag"] = events["post_permlink"].apply(lambda x: posts.loc[x]["last_tag"] if x in posts.index else "")
  events["topic"] = events["post_permlink"].apply(lambda x: posts.loc[x]["topic"] if x in posts.index else 0)
  events["topic_probability"] = events["post_permlink"].apply(lambda x: posts.loc[x]["topic_probability"] if x in posts.index else 0)
  popularity = events.groupby("post_permlink").describe()["like"]["count"]
  events["popularity"] = events["post_permlink"].apply(lambda x: popularity.loc[x])
  events["popularity_coefficient"] = quantile_transform(events["popularity"].reshape(-1, 1), output_distribution="normal", copy=True).reshape(-1)
  events["time"] = events["post_permlink"].apply(lambda x: posts.loc[x]["created"].value if x in posts.index else np.nan)
  events["time_coefficient"] = quantile_transform(events["time"].fillna(events["time"].median()).reshape(-1, 1), output_distribution="normal", copy=True).reshape(-1)
  return events

def create_mapping(series):
  mapping = {}
  for (idx, mid) in enumerate(np.unique(series)):
    mapping[mid] = idx
  return mapping

def create_ffm_dataset(events, mapping=None):
  if not mapping:
    uid_to_idx = create_mapping(events["user_id"])
    pid_to_idx = create_mapping(events["post_permlink"])
    aid_to_idx = create_mapping(events["author"])
    parid_to_idx = create_mapping(events["parent_permlink"])
    ftgid_to_idx = create_mapping(events["first_tag"])
    ltgid_to_idx = create_mapping(events["last_tag"])
  else:
    uid_to_idx = mapping["uid_to_idx"]
    pid_to_idx = mapping["pid_to_idx"]
    aid_to_idx = mapping["aid_to_idx"]
    parid_to_idx = mapping["parid_to_idx"]
    ftgid_to_idx = mapping["ftgid_to_idx"]
    ltgid_to_idx = mapping["ltgid_to_idx"]

  result = []
  for _, event in events.iterrows():
    result.append([
      (0, uid_to_idx.get(event["user_id"], max(uid_to_idx.values()) + 1), 1),
      (1, pid_to_idx.get(event["post_permlink"], max(pid_to_idx.values()) + 1), 1),
      (2, aid_to_idx.get(event["author"], max(aid_to_idx.values()) + 1), 1),
      (3, parid_to_idx.get(event["parent_permlink"], max(parid_to_idx.values()) + 1), 1),
      (4, ftgid_to_idx.get(event["first_tag"], max(ftgid_to_idx.values()) + 1), 1),
      (5, ltgid_to_idx.get(event["last_tag"], max(ltgid_to_idx.values()) + 1), 1),
      (6, event["topic"], event["topic_probability"]),
      (7, 1, event["time_coefficient"]),
      (8, 1, event["popularity_coefficient"]),
    ])
  return {
    'uid_to_idx': uid_to_idx,
    'pid_to_idx': pid_to_idx,
    'aid_to_idx': aid_to_idx,
    'parid_to_idx': parid_to_idx,
    'ftgid_to_idx': ftgid_to_idx,
    'ltgid_to_idx': ltgid_to_idx,
  }, result, (events["like"] > 0.5).tolist()

def build_model(train_X, train_y, test_X, test_y):
  train_ffm_data = ffm.FFMData(train_X, train_y)
  test_ffm_data = ffm.FFMData(test_X, test_y)

  model = ffm.FFM(**MODEL_PARAMETERS)
  model.init_model(train_ffm_data)

  for i in range(ITERATIONS):
    model.iteration(train_ffm_data)
  return model, roc_auc_score(train_y, model.predict(train_ffm_data)), roc_auc_score(test_y, model.predict(test_ffm_data))

def train(raw_events, database_url, database):
  print("Prepare user events...")
  user_events = get_user_events(raw_events)
  print("Prepare events...")
  events = get_events(user_events)
  print("Prepare posts...")
  posts = get_posts(database_url, database)
  print("Extend events...")
  extend_events(events, posts)
  print("Create ffm dataset...")
  mappings, X, y = create_ffm_dataset(events)
  train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)
  print("Build model...")
  model, train_auc_roc, test_auc_roc = build_model(train_X, train_y, test_X, test_y)
  print(train_auc_roc)
  print(test_auc_roc)
  model.save_model("./model.bin")
  joblib.dump(mappings, "./mappings.pkl")

if (__name__ == "__main__"):
  raw_events = pd.read_csv(sys.argv[1], names=["id", "event_type", "value", "user_id", "refurl", "status", "created_at"])
  train(raw_events, sys.argv[2], sys.argv[3])
