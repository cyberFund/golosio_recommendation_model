from model.train import get_posts
import re
from bs4 import BeautifulSoup
import html
from markdown import markdown
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from pymystem3 import Mystem
import numpy as np
from nltk.probability import FreqDist
from gensim import corpora


stopwords_list = stopwords.words('russian')
tokenizer = RegexpTokenizer(r'\w+')
stemmer = Mystem()

WORD_LENGTH_QUANTILE = 10
TEXT_LENGTH_QUANTILE = 66
HIGH_WORD_FREQUENCY_QUANTILE = 99
LOW_WORD_FREQUENCY_QUANTILE = 60

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

def remove_short_words(texts):
  word_lengths = [len(item) for sublist in texts for item in sublist]
  word_length_quantile = np.percentile(np.array(word_lengths), WORD_LENGTH_QUANTILE)
  return [[word for word in text if len(word) >= word_length_quantile] for text in texts]

def remove_short_texts(texts):
  text_lengths = [len(text) for text in texts]
  text_length_quantile = np.percentile(np.array(text_lengths), TEXT_LENGTH_QUANTILE)
  return [text for text in texts if len(text) >= text_length_quantile]

def remove_high_frequent_words(texts):
  dictionary = FreqDist([item for sublist in texts for item in sublist])
  word_frequencies = list(dictionary.values())
  high_word_frequency_quantile = np.percentile(np.array(word_frequencies), HIGH_WORD_FREQUENCY_QUANTILE)
  return [[word for word in text if dictionary[word] < high_word_frequency_quantile] for text in texts]

def remove_low_frequent_words(texts):
  dictionary = FreqDist([item for sublist in texts for item in sublist])
  word_frequencies = list(dictionary.values())
  low_word_frequency_quantile = np.percentile(np.array(word_frequencies), LOW_WORD_FREQUENCY_QUANTILE)
  return [[word for word in text if dictionary[word] >= low_word_frequency_quantile] for text in texts]

def prepare_posts(posts):
  posts = [prepare_post(post) for post in posts]
  posts = remove_short_words(posts)
  posts = remove_high_frequent_words(posts)
  posts = remove_low_frequent_words(posts)
  return remove_short_texts(posts)

def create_dictionary(texts):
  dictionary = corpora.Dictionary(texts)
  dictionary.save('golos-corpora.dict')