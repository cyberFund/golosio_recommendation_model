import re
from bs4 import BeautifulSoup
import html
from markdown import markdown
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from pymystem3 import Mystem
import numpy as np
from pymongo import MongoClient
import pdb
from tqdm import *

stopwords_list = stopwords.words('russian')
tokenizer = RegexpTokenizer(r'\w+')
stemmer = Mystem()

def preprocess_posts(posts, include_all_tags=False):
  posts["post_permlink"] = "@" + posts["author"] + "/" + posts["permlink"]
  if include_all_tags:
    posts["tags"] = posts["json_metadata"].apply(lambda x: x["tags"] if (type(x) is dict and "tags" in x.keys()) else [])
  else:
    posts["first_tag"] = posts["json_metadata"].apply(lambda x: x["tags"][0] if (type(x) is dict and "tags" in x.keys() and len(x["tags"])) else "")
    posts["last_tag"] = posts["json_metadata"].apply(lambda x: x["tags"][-1] if (type(x) is dict and "tags" in x.keys() and len(x["tags"])) else "")
  return posts.drop(["json_metadata", "_id"], axis=1)

def topics_to_vector(topics, n_topics=100):
  vector = np.zeros(n_topics)
  for topic, probability in topics:
    vector[topic] = probability
  return vector

def remove_usernames(post):
  return re.sub('@\w+\s', '', post)

def remove_html_tags(post):
  return BeautifulSoup(post, "lxml").get_text()

def unescape_html_tags(post):
  return html.unescape(post)

def convert_markdown(post):
  return markdown(post)

def remove_stopwords(words):
  return [word for word in words if word not in stopwords_list]

def split_post(post):
  return tokenizer.tokenize(post)

def lemmatize(words):
  return [stemmer.lemmatize(word)[0] for word in words]

def prepare_post(post):
  post = post.lower()
  post = unescape_html_tags(post)
  post = convert_markdown(post)
  post = remove_html_tags(post)
  post = remove_usernames(post)
  words = split_post(post)
  words = remove_stopwords(words)
  return lemmatize(words)

def save_topics(url, database, posts, texts, model, dictionary):
  client = MongoClient(url)
  db = client[database]
  posts["prepared_body"] = texts
  for index in tqdm(posts.index):
    post = posts.loc[index]
    post_topics = model.get_document_topics(dictionary.doc2bow(post["prepared_body"]))
    vector = topics_to_vector(post_topics, n_topics=100)
    topic = int(np.argmax(vector))
    topic_probability = float(np.max(vector))
    db.comment.update_one({'_id': post["post_permlink"][1:]}, {'$set': {'topic': topic, 'topic_probability': topic_probability}})