from pymongo import MongoClient, DESCENDING, ASCENDING
from golosio_recommendation_model.config import config
from golos import Steem

STEP_BACKWARD_SIZE = 20
STEP_FORWARD_SIZE = 10

def get_node():
  return Steem([config["node_url"]])

def get_database():
  client = MongoClient(config["database_url"])
  database = client[config["database_name"]]
  return database
  
def no_posts():
  database = get_database()
  return database.comment.count() == 0

def find_newest_post():
  database = get_database()
  newest_posts = database.comment.find({}, {"_id": True}).sort([("created", DESCENDING)])
  return newest_posts[0]["_id"]

def find_oldest_post():
  database = get_database()
  oldest_posts = database.comment.find({}, {"_id": True}).sort([("created", ASCENDING)])
  return oldest_posts[0]["_id"]

def get_newest_post_as_consistent():
  database = get_database()
  newest_post = find_newest_post()
  database.comment.update({"_id": newest_post}, {"$set": {"consistent": True}})
  return newest_post

def find_newest_consistent_post():
  database = get_database()
  newest_consistent_posts = database.comment.find({"consistent": True}, {"_id": True}).sort([("created", ASCENDING)])
  if newest_consistent_posts.count() > 0:
    return newest_consistent_posts[0]["_id"]
  else:
    return get_newest_post_as_consistent()

def save_posts(posts):
  database = get_database()
  posts_for_database = [{
    "_id": post.identifier[1:],
    "permlink": post.permlink,
    "author": post.author,
    "parent_permlink": post.parent_permlink,
    "created": post.created,
    "json_metadata": post.json_metadata,
    "body": post.body,
    "depth": 0
  } for post in posts]
  posts_in_database = database.comment.find({
    '_id' : {
      '$in' : [post["_id"] for post in posts_for_database]
    }
  })
  posts_in_database = set(post["_id"] for post in posts_in_database)
  filtered_posts_for_database = [post for post in posts_for_database if post["_id"] not in posts_in_database]
  if len(filtered_posts_for_database):
    database.comment.insert(filtered_posts_for_database)

def do_initial_step():
  database = get_database()
  node = get_node()
  posts = node.get_posts(limit=STEP_BACKWARD_SIZE, sort='created') # Sort by created
  save_posts(posts)

def do_step_backward():
  do_initial_step()