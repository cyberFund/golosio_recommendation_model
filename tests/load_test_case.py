import unittest
import os
import numpy as np
from functools import wraps
from time import time
import requests
from pymongo import MongoClient
import pandas as pd
import pdb

SAMPLE_SIZE = 100

def timing(f):
  @wraps(f)
  def wrapper(*args, **kwargs):
    start = time()
    result = f(*args, **kwargs)
    end = time()
    print('Elapsed time: {}'.format(end-start))
    print('Average time: {}'.format((end-start) / SAMPLE_SIZE))
    return result
  return wrapper

class LoadTestCase(unittest.TestCase):
  @timing
  def do_requests(self, urls):
    for url in urls:
      requests.get(url, verify=False)

  def get_posts(self):
    client = MongoClient("localhost:27017")
    db = client["steemdb_1"]
    posts = pd.DataFrame(list(db.comment.find(
      {
        'depth': 0,
      }, {
        '_id': 1,
      }
    ).limit(10000)))
    return list(set(posts["_id"].tolist()))

  def get_users(self):
    client = MongoClient("localhost:27017")
    db = client["steemdb_1"]
    events = pd.DataFrame(list(db.event.find(
      {
      }, {
        'user_id': 1,
      }
    ).limit(10000)))
    return list(set(events["user_id"].tolist()))

  def test_similar(self):
    # Get all posts
    posts = self.get_posts()
    # Get sample
    posts = np.random.choice(posts, size=SAMPLE_SIZE, replace=True)
    posts = ["http://localhost:8080/similar?permlink=@" + post for post in posts]
    # Do requests
    self.do_requests(posts)
    pass

  def test_recommendations(self):
    # Get all users
    users = self.get_users()
    # Get sample for N users
    users = np.random.choice(users, size=SAMPLE_SIZE, replace=True)
    users = ["http://localhost:8080/recommendations?user=" + user for user in users]
    # Do requests
    self.do_requests(users)