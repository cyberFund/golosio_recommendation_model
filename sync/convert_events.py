from pymongo import MongoClient, DESCENDING
import datetime as dt
import pandas as pd
from tqdm import *
import pdb
import dask.dataframe as dd
import sys

HOURS_LIMIT = 14 * 24 # Time window for recommended posts

def get_last_event_date(url, database):
  client = MongoClient(url)
  db = client[database]
  last_event = db.event.find(
    {
    }, {
      'created_at': 1,
    }
  ).sort([("created_at", DESCENDING)]).limit(1)[0]
  return last_event['created_at']

def get_raw_events(url, database): 
  date = get_last_event_date(url, database) - dt.timedelta(hours=HOURS_LIMIT)
  client = MongoClient(url) 
  db = client[database] 
  events = pd.DataFrame(list(db.raw_event.find( 
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

def parse_refurl(url):
  return "/".join(url.split("/")[4:])

def parse_recommendations(urls):
  return ["@" + url[1:-1] for url in urls[1:-1].split(",") if len(url) > 0]

def remove_last_events(url, database)
  client = MongoClient(url)
  db = client[database]
  db.drop_collection('event')

def prepare_raw_events(raw_events):
  raw_events["refurl"] = raw_events["refurl"].astype(str)
  raw_events["value"] = raw_events["value"].astype(str)
  raw_events["user_id"].fillna("\\N", inplace=True)
  return raw_events[raw_events["user_id"].astype(str) != "\\N"]

def get_user_events(raw_events):
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

def get_coefficient(user_events, user, post):
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

def convert_dataframe(raw_events):
  raw_events = prepare_raw_events(raw_events)
  user_events = get_user_events(raw_events)
  events = get_events(user_events)
  return events

def save_events(url, database, events):
  client = MongoClient(url)
  db = client[database]
  db.event.insert_many(events.to_dict('records'))

def convert_events(database_url, database_name):
  raw_events = get_raw_events(database_url, database_name)
  events = convert_dataframe(raw_events)
  remove_last_events(database_url, database_name)
  save_events(database_url, database_name, events_path)

if (__name__ == "__main__"):
  convert_events(sys.argv[1], sys.argv[2])