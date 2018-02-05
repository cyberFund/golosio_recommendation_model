import model.predict as ffm
import model.train as train_ffm
import unittest
from pymongo import MongoClient
import pandas as pd
import datetime as dt

class PredictTestCase(unittest.TestCase):
  def create_raw_events_dataframe(self, user_id=1):
    raw_events = pd.read_csv("./tests/raw_events.csv")
    raw_events["user_id"] = user_id
    return raw_events

  def create_big_raw_events_dataframe(self):
    return pd.concat([self.create_raw_events_dataframe(user_id=i) for i in range(10)])

  def create_posts_dataframe(self):
    return pd.read_csv("./tests/posts.csv")

  def create_events_dataframe(self):
    return pd.read_csv("./tests/events.csv")

  def create_recommendations_dataframe(self):
    return pd.read_csv("./tests/recommendations.csv")

  def create_big_events_dataframe(self):
    return pd.concat([self.create_events_dataframe() for _ in range(10)])

  def test_get_new_posts(self):
    client = MongoClient("localhost:27017")
    db = client["steemdb_test"]
    db.drop_collection('comment')
    test_posts = [{
      "permlink": "link",
      "author": "someauthor", 
      "parent_permlink": "somelink",
      "json_metadata": {
        "tags": ["tag1", "tag2"]
      },
      "depth": 0,
      "created": pd.Timestamp.now().round('60min'),
      "topic": 0,
      "topic_probability": 0.9
    }, {
      "permlink": "link2",
      "author": "someauthor", 
      "parent_permlink": "somelink",
      "json_metadata": {
        "tags": ["tag1", "tag2"]
      },
      "depth": 0,
      "created": (pd.Timestamp.now() - dt.timedelta(days=5)).round('60min'),
      "topic": 0,
      "topic_probability": 0.9
    }]
    db.comment.insert_many(test_posts)
    posts = ffm.get_new_posts("localhost:27017", "steemdb_test")
    self.assertCountEqual([p for p in posts.columns], ["post_permlink", "author", "parent_permlink", "first_tag", "last_tag", "created", "topic", "topic_probability"])
    self.assertCountEqual(posts.loc[0].tolist(), ["@someauthor/link", "someauthor", "somelink", "tag1", "tag2", pd.Timestamp.now().round('60min'), 0, 0.9])
    self.assertEqual(posts.shape[0], 1)

  def test_create_dataset(self):
    posts = self.create_posts_dataframe()
    users = [1, 2, 3]
    events = ffm.create_dataset(posts, users)
    assert events.shape[0] == posts.shape[0] * len(users)
    assert "time_coefficient" in events.columns 
    self.assertCountEqual(events[events["user_id"] == 1]["post_permlink"], posts["post_permlink"])

  def test_save_recommendations(self):
    client = MongoClient("localhost:27017")
    recommendations = self.create_recommendations_dataframe()
    db = client["steemdb_test"]
    db.drop_collection('recommendation')
    ffm.save_recommendations(recommendations, "localhost:27017", "steemdb_test")
    valid_recommendations = [{
      'user_id': 1,
      'post_permlink': 'somelink1',
      'prediction': 0.1
    }, {
      'user_id': 1,
      'post_permlink': 'somelink2',
      'prediction': 0.3
    }, {
      'user_id': 2,
      'post_permlink': 'somelink1',
      'prediction': 0.1
    }]
    for recommendation in valid_recommendations:
      assert db.recommendation.find(recommendation)[0]

  def test_predict(self):
    client = MongoClient("localhost:27017")
    db = client["steemdb_test"]
    db.drop_collection('comment')
    db.drop_collection('recommendation')
    test_posts = [{
      "permlink": "somelink1",
      "author": "author", 
      "parent_permlink": "somelink",
      "json_metadata": {
        "tags": ["tag1", "tag2"]
      },
      "depth": 0,
      "created": pd.Timestamp.now().round('60min'),
      "topic": 0,
      "topic_probability": 0.9
    }, {
      "permlink": "somelink2",
      "author": "author", 
      "parent_permlink": "somelink",
      "json_metadata": {
        "tags": ["tag1", "tag2"]
      },
      "depth": 0,
      "created": (pd.Timestamp.now() - dt.timedelta(days=5)).round('60min'),
      "topic": 0,
      "topic_probability": 0.9
    }]
    db.comment.insert_many(test_posts)
    raw_events = self.create_big_raw_events_dataframe()
    train_ffm.train(raw_events, "localhost:27017", "steemdb_test")
    ffm.predict(raw_events, "localhost:27017", "steemdb_test")
    assert len(list(db.recommendation.find({}))) > 0