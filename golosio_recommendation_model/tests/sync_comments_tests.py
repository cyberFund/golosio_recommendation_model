import os
os.environ["GOLOSIO_DATABASE_NAME"] = "test_golos_database"
os.environ["GOLOSIO_NODE_URL"] = 'wss://ws.golos.io'
import unittest
from golosio_recommendation_model.sync.sync_comments import *
from golosio_recommendation_model.config import config
from pymongo import MongoClient, DESCENDING, ASCENDING
from datetime import datetime, timedelta
from golos import Steem

# Set env variables
# test_golos_comments

class SyncCommentsTestCase(unittest.TestCase):
  def setUp(self):
    self.node = Steem([config["node_url"]])
    self.client = MongoClient(config["database_url"])
    self.client.drop_database(config["database_name"])
    self.database = self.client[config["database_name"]]

  def test_no_posts(self):
    first_result = no_posts()
    self.database.comment.insert_one({'test': 1})
    second_result = no_posts()
    assert first_result
    assert not second_result

  def insert_posts(self):
    self.database.comment.insert_one({'_id': 1, "created": datetime.now() - timedelta(1)})
    self.database.comment.insert_one({'_id': 2, "created": datetime.now()})
    self.database.comment.insert_one({'_id': 3, "created": datetime.now() - timedelta(2)})

  def test_find_newest_post(self):
    self.insert_posts()
    newest_post = find_newest_post()
    assert newest_post == 2

  def test_find_oldest_post(self):
    self.insert_posts()
    oldest_post = find_oldest_post()
    assert oldest_post == 3

  def test_get_newest_post_as_consistent(self):
    self.insert_posts()
    newest_post = get_newest_post_as_consistent()
    post_in_database = self.database.comment.find_one({"_id": newest_post})
    assert newest_post == 2
    assert post_in_database["consistent"]

  def test_find_newest_consistent_post(self):
    self.insert_posts()
    self.database.comment.update({"_id": 1}, {"$set": {"consistent": True}})
    newest_consistent_post = find_newest_consistent_post()
    assert newest_consistent_post == 1

  def test_find_newest_consistent_post_initially(self):
    self.insert_posts()
    newest_consistent_post = find_newest_consistent_post()
    post_in_database = self.database.comment.find_one({"_id": newest_consistent_post})
    assert newest_consistent_post == 2
    assert post_in_database["consistent"]

  def test_save_posts(self):
    posts = self.node.get_posts(limit=10, sort='trending')
    save_posts(posts)
    posts_in_database = self.database.comment.find({})
    for index, post_in_database in enumerate(posts_in_database):
      post = posts[index]
      self.assertCountEqual(post_in_database, {
        "_id": post.identifier[1:],
        "permlink": post.permlink,
        "author": post.author,
        "parent_permlink": post.parent_permlink,
        "created": post.created,
        "json_metadata": post.json_metadata,
        "body": post.body,
        "depth": 0
      })

  def test_unique_save_posts(self):
    posts = self.node.get_posts(limit=10, sort='trending')
    save_posts(posts)
    save_posts(posts)
    number_of_posts_in_database = self.database.comment.count({})
    assert number_of_posts_in_database == len(posts)

  def test_set_index(self):
    set_index()
    indices = self.database.comment.index_information()
    indices = [value["key"][0] for key, value in indices.items()]
    assert ("created", DESCENDING) in indices

  def test_initial_step(self):
    do_initial_step()
    number_of_posts_in_database = self.database.comment.count({})
    consistent_posts = list(self.database.comment.find({"consistent": True}))
    newest_post = self.database.comment.find({}).sort([("created", DESCENDING)])[0]
    assert number_of_posts_in_database == STEP_BACKWARD_SIZE

  def test_step_backward(self):
    last_post = do_step_backward("khorunzha/uznai-o-svoem-potenciale")
    last_post_in_database = self.database.comment.find_one({"_id": last_post})
    number_of_extracted_posts = self.database.comment.count({"created": {
      "$gte": last_post_in_database["created"]
    }})
    assert last_post == "sinilga/ya-uzhe-dykhane-beregu"
    assert number_of_extracted_posts == STEP_BACKWARD_SIZE

  def test_step_forward(self):
    last_consistent_post, last_post = do_step_forward("hipster/post-dobra", "sashapoplavskiy/skyfchain-ico")
    last_post_in_database = self.database.comment.find_one({"_id": last_post})
    number_of_extracted_posts = self.database.comment.count({"created": {
      "$gte": last_post_in_database["created"]
    }})
    assert last_post == "markonly/localcoinswap-kyc-teper-ne-nuzhen-ico-live"
    assert last_consistent_post == "hipster/post-dobra"
    assert number_of_extracted_posts == STEP_FORWARD_SIZE

  def test_final_step_forward(self):
    self.database.comment.insert_one({'_id': "some/new-post", "created": datetime.now()})
    previous_post = "sashapoplavskiy/skyfchain-ico"
    last_consistent_post, last_post = do_step_forward("markonly/localcoinswap-kyc-teper-ne-nuzhen-ico-live", previous_post)
    number_of_posts = self.database.comment.count({})
    newest_post_in_database = self.database.comment.find_one({"_id": "some/new-post"})
    assert last_consistent_post == "some/new-post"
    assert not last_post
    assert newest_post_in_database["consistent"] == True

  def test_sync_comments(self):
    do_initial_step()
    oldest_post = find_oldest_post()
    oldest_post_in_database = self.database.comment.find_one({"_id": oldest_post})
    self.database.comment.remove({"_id": {"$ne": oldest_post}})
    sync_comments(max_iterations=1)
    number_of_newest_posts = self.database.comment.count({"created": {"$gt": oldest_post_in_database["created"]}})
    number_of_oldest_posts = self.database.comment.count({"created": {"$lt": oldest_post_in_database["created"]}})
    assert number_of_oldest_posts == STEP_BACKWARD_SIZE
    assert number_of_newest_posts == STEP_FORWARD_SIZE

  def test_initial_sync_comments(self):
    sync_comments(max_iterations=0)
    number_of_posts = self.database.comment.count({})
    indices = self.database.comment.index_information()
    indices = [value["key"][0] for key, value in indices.items()]
    assert number_of_posts == STEP_BACKWARD_SIZE
    assert ("created", DESCENDING) in indices
