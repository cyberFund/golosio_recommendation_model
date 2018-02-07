import re
from bs4 import BeautifulSoup
import html
from markdown import markdown
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from pymystem3 import Mystem
import numpy as np
from pymongo import MongoClient, DESCENDING
from tqdm import *
import logging
from time import sleep
from functools import wraps
import pandas as pd
import os
from golosio_recommendation_model.config import config

logging.basicConfig(filename=config['log_path'], format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

stopwords_list = stopwords.words('russian')
tokenizer = RegexpTokenizer(r'\w+')
stemmer = Mystem()

def get_events(url, database): 
  """
    Function to get latest events from a database
  """
  client = MongoClient(url) 
  db = client[database] 
  events = pd.DataFrame(list(db.event.find( 
    {
    }, { 
      'user_id': 1,  
      'post_permlink' : 1, 
      'like' : 1, 
    } 
  ))) 
  return events 

def get_posts(url, database, events, filter_options={}):
  """
    Function to get all posts from a database
  """
  client = MongoClient(url)
  db = client[database]
  posts = pd.DataFrame(list(db.comment.find(
    {**({
      '_id' : {
        '$in' : list(set(events["post_permlink"].apply(lambda x: x[1:])))
      },
      'depth': 0,
    }), **filter_options}, {
      'permlink': 1,
      'author': 1, 
      'parent_permlink': 1,
      'created': 1,
      'json_metadata': 1,
      'similar_posts': 1,
      'similar_distances': 1,
      'inferred_vector': 1,
      'body': 1,
      'prepared_body': 1
    }
  )), columns=["_id", "permlink", "author", "parent_permlink", "created", "json_metadata", "similar_posts", "similar_distances", "inferred_vector", "body", "prepared_body"])
  return posts

def get_last_post_date(url, database):
  client = MongoClient(url)
  db = client[database]
  last_post = db.comment.find(
    {
      'permlink' : {'$exists' : True},
      'depth': 0,
    }, {
      'created': 1,
    }
  ).sort([("created", DESCENDING)]).limit(1)[0]
  return last_post['created']

def preprocess_posts(posts, include_all_tags=False):
  """
    Function to add full permlink and tags for each post in dataframe extracted from mongo
  """
  posts["post_permlink"] = "@" + posts["author"] + "/" + posts["permlink"]
  if include_all_tags:
    posts["tags"] = posts["json_metadata"].apply(lambda x: x["tags"] if (type(x) is dict and "tags" in x.keys()) else [])
  else:
    posts["first_tag"] = posts["json_metadata"].apply(lambda x: x["tags"][0] if (type(x) is dict and "tags" in x.keys() and len(x["tags"])) else "")
    posts["last_tag"] = posts["json_metadata"].apply(lambda x: x["tags"][-1] if (type(x) is dict and "tags" in x.keys() and len(x["tags"])) else "")
  return posts.drop(["json_metadata", "_id"], axis=1)

def remove_usernames(post):
  """
    Function to remove usernames from a text
  """
  return re.sub('@\w+\s', '', post)

def remove_html_tags(post):
  """
    Function to remove html tags from a text
  """
  return BeautifulSoup(post, "lxml").get_text()

def unescape_html_tags(post):
  """
    Function to unescape html in a text
  """
  return html.unescape(post)

def convert_markdown(post):
  """
    Function to convert markdown to a html in a text
  """
  return markdown(post)

def remove_stopwords(words):
  """
    Function to remove stopwords from a list
  """
  return [word for word in words if word not in stopwords_list]

def split_post(post):
  """
    Function to tokenize post
  """
  return tokenizer.tokenize(post)

def lemmatize(words):
  """
    Function to lemmatize each word in a post
  """
  return [stemmer.lemmatize(word)[0] for word in words]

def prepare_post(post):
  """
    Function to prepare a post for LDA and Doc2Vec processes
  """
  if ("prepared_body" in post) and (not np.isnan(post["prepared_body"])):
    return post["prepared_body"]
  post = post["body"].lower()
  post = unescape_html_tags(post)
  post = convert_markdown(post)
  post = remove_html_tags(post)
  post = remove_usernames(post)
  words = split_post(post)
  words = remove_stopwords(words)
  return lemmatize(words)

def log(model, message):
  """
    Function to print given message to a log
  """
  logging.warning(model + ": " + str(message))
  print(model + ": " + str(message))

def error_log(model):
  def error_log_decorator(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
      try:
        f(*args, **kwargs)
      except Exception as e:
        log(model, "An exception appeared")
        logging.exception(e)
        print(e)
      else:
        log(model, "Finished successfully")
    return wrapper
  return error_log_decorator

def wait_between_iterations():
  sleep(5 * 60)