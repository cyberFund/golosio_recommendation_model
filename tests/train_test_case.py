import unittest
import pdb
import model.train as ffm
import pandas as pd
from sklearn.metrics import roc_auc_score
import ffm as ffm_utils
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
from sklearn.externals import joblib
from sklearn.preprocessing import quantile_transform
import os

class TrainTestCase(unittest.TestCase):
  def create_raw_events_dataframe(self, user_id=1):
    raw_events = pd.read_csv("./tests/raw_events.csv")
    raw_events["user_id"] = user_id
    return raw_events

  def create_big_raw_events_dataframe(self):
    return pd.concat([self.create_raw_events_dataframe(user_id=i) for i in range(10)])

  def create_user_events_dataframe(self):
    return pd.read_csv("./tests/user_events.csv")

  def create_events_dataframe(self):
    return pd.read_csv("./tests/events.csv")

  def create_posts_dataframe(self):
    return pd.read_csv("./tests/posts.csv")

  def create_big_events_dataframe(self):
    return pd.concat([self.create_events_dataframe() for _ in range(10)])

  def test_parse_refurl(self):
    self.assertEqual(ffm.parse_refurl("https://golos.io/permlink/@author/somelink1"), "@author/somelink1")

  def test_parse_recommendation(self):
    self.assertCountEqual(ffm.parse_recommendations("[-author/somelink1-,-author/somelink2-]"), ["@author/somelink1", "@author/somelink2"])

  def test_get_user_events(self):
    raw_events = self.create_raw_events_dataframe()
    user_events = ffm.get_user_events(raw_events)
    user_event = user_events.iloc[0]
    self.assertCountEqual(user_event["views"], ["@author/somelink1", "@author/somelink2"])
    self.assertCountEqual(user_event["votes"], ["@author/somelink2"])
    self.assertCountEqual(user_event["comments"], ["@author/somelink2"])
    self.assertCountEqual(user_event["recommendations"], ["@author/somelink1", "@author/somelink2", "@author/somelink3"])
    assert user_event["user"] == 1

  def test_get_events(self):
    raw_events = self.create_raw_events_dataframe()
    user_events = ffm.get_user_events(raw_events)
    events = ffm.get_events(user_events)
    self.assertCountEqual(events["user_id"], [1] * 3)
    self.assertCountEqual(events["post_permlink"], ["@author/somelink1", "@author/somelink2", "@author/somelink3"])
    self.assertCountEqual(events[events["post_permlink"] == "@author/somelink2"]["like"], [1])
    self.assertCountEqual(events[events["post_permlink"] == "@author/somelink1"]["like"], [0.7])
    self.assertCountEqual(events[events["post_permlink"] == "@author/somelink3"]["like"], [-1])    

  def test_get_posts(self):
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
    }]
    db.comment.insert_many(test_posts)
    posts = ffm.get_posts("localhost:27017", "steemdb_test")
    self.assertCountEqual([p for p in posts.columns], ["post_permlink", "author", "parent_permlink", "first_tag", "last_tag", "created", "topic", "topic_probability"])
    self.assertCountEqual(posts.loc[0].tolist(), ["@someauthor/link", "someauthor", "somelink", "tag1", "tag2", pd.Timestamp.now().round('60min'), 0, 0.9])

  def test_extend_events(self):
    raw_events = self.create_raw_events_dataframe()
    user_events = ffm.get_user_events(raw_events)
    posts = self.create_posts_dataframe()
    events = ffm.get_events(user_events)
    ffm.extend_events(events, posts)
    event = events[events["post_permlink"] == "@author/somelink1"].iloc[0]
    self.assertEqual(event["parent_permlink"], "somelink")
    self.assertEqual(event["author"], "author")
    self.assertEqual(event["first_tag"], "tag1")
    self.assertEqual(event["last_tag"], "tag2")
    self.assertEqual(event["topic"], 0)
    self.assertEqual(event["topic_probability"], 0.9)
    self.assertEqual(event["popularity"], 1)
    # Test for popularity and time coefficient
    event = events[events["post_permlink"] == "@author/somelink2"].iloc[0]
    assert event["parent_permlink"] == ""

  def test_extend_events_with_nan(self):
    raw_events = self.create_raw_events_dataframe()
    user_events = ffm.get_user_events(raw_events)
    posts = self.create_posts_dataframe()
    events = ffm.get_events(user_events)
    ffm.extend_events(events, posts)
    event = events[events["post_permlink"] == "@author/somelink2"].iloc[0]
    self.assertEqual(event["topic"], 0)
    self.assertEqual(event["topic_probability"], 0)

  def test_create_mapping(self):
    uid_to_idx = ffm.create_mapping([10, 20, 12])
    assert uid_to_idx[20] != uid_to_idx[10]

  def test_create_ffm_dataset(self):
    events = self.create_events_dataframe()
    mappings, ffm_data_X, ffm_data_y = ffm.create_ffm_dataset(events)
    assert [
      (0, mappings['uid_to_idx'][13], 1), 
      (1, mappings['pid_to_idx']["somepost"], 1),
      (2, mappings['aid_to_idx']["someauthor"], 1),
      (3, mappings['parid_to_idx']["somelink"], 1),
      (4, mappings['ftgid_to_idx']["tag1"], 1),
      (5, mappings['ltgid_to_idx']["tag2"], 1),
      (6, 0, 0.9),
      (7, 1, 0.8),
      (8, 1, 0.6),
    ] in ffm_data_X
    assert [0, 1] == ffm_data_y

  def test_create_ffm_dataset_with_existed_mapping(self):
    events = self.create_events_dataframe()
    mappings, ffm_data_X, ffm_data_y = ffm.create_ffm_dataset(events)
    events.iloc[0, 1] = 100
    new_mappings, ffm_data_X, ffm_data_y = ffm.create_ffm_dataset(events, mappings)
    assert new_mappings == mappings
    assert (0, max(*mappings['uid_to_idx'].values()) + 1, 1) in ffm_data_X[0]

  def test_build_model(self):
    events = self.create_big_events_dataframe()
    train, test = train_test_split(events, test_size=0.5)
    train_mappings, train_ffm_data_X, train_ffm_data_y = ffm.create_ffm_dataset(train)
    test_mappings, test_ffm_data_X, test_ffm_data_y = ffm.create_ffm_dataset(test)
    train_ffm_data = ffm_utils.FFMData(train_ffm_data_X, train_ffm_data_y)
    test_ffm_data = ffm_utils.FFMData(test_ffm_data_X, test_ffm_data_y)
    model, train_roc_auc, test_roc_auc = ffm.build_model(train_ffm_data_X, train_ffm_data_y, test_ffm_data_X, test_ffm_data_y)
    assert train_roc_auc == roc_auc_score(train_ffm_data_y, model.predict(train_ffm_data)) 
    assert test_roc_auc == roc_auc_score(test_ffm_data_y, model.predict(test_ffm_data)) 

  def test_train(self):
    client = MongoClient("localhost:27017")
    db = client["steemdb_test"]
    db.drop_collection('comment')
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
    }]
    db.comment.insert_many(test_posts)
    raw_events = self.create_big_raw_events_dataframe()
    os.remove("model.bin")
    os.remove("mappings.pkl")
    ffm.train(raw_events, "localhost:27017", "steemdb_test")
    model = ffm_utils.read_model('model.bin')
    mappings = joblib.load('mappings.pkl')
    assert mappings['uid_to_idx']
    assert model.predict