import model.predict as ffm

class PredictTestCase(unittest.TestCase):
  def create_events_dataframe(self):
    return pd.read_csv("./tests/events.csv")

  def create_big_events_dataframe(self):
    return pd.concat([self.create_events_dataframe() for _ in range(10)])

  def test_get_new_posts():
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
    new_posts = ffm.get_new_posts()
    self.assertCountEqual([p for p in posts.columns], ["post_permlink", "author", "parent_permlink", "first_tag", "last_tag", "created", "topic", "topic_probability"])
    self.assertCountEqual(posts.loc[0].tolist(), ["@someauthor/link", "someauthor", "somelink", "tag1", "tag2", pd.Timestamp.now().round('60min'), 0, 0.9])

  def test_load_model():
    # events = self.create_big_events_dataframe()
    # mappings, X, y = ffm.create_ffm_dataset(events)
    # ffm_data = ffm_utils.FFMData(X, y)
    # model = ffm_utils.FFM()
    # model.init_model(ffm_data)
    # ffm.save_model(mappings, model)
    # assert joblib.load("mappings.pkl")['uid_to_idx']
    # assert ffm_utils.read_model("model.bin").predict
    # Create model and mapping, save it to a disc
    # Load model and mapping
    # Check methods
    pass

  def test_create_dataset():

    pass

  def test_save_recommendations():
    pass
