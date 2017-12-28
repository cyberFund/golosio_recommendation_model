from model import utils
from gensim import models, corpora

def get_new_posts(url, database):
  date = dt.datetime.now() - dt.timedelta(hours=5)
  posts = pd.DataFrame()
  client = MongoClient(url)
  db = client[database]
  posts = pd.DataFrame(list(db.comment.find(
    {
      'permlink' : {'$exists' : True},
      'depth': 0,
      'created': {'$gte': date}
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

def run_lda(database_url, database_name):
  posts = get_new_posts(database_url, database_name)
  dictionary = corpora.Dictionary.load('golos-corpora.dict')
  model = models.LdaMulticore.load('golos.lda_model')
  utils.save_topics(database_url, database_name, posts, model, dictionary)