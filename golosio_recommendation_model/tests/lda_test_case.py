import unittest
from pymongo import MongoClient
import pandas as pd
import model.lda as lda
from gensim import corpora

class LdaTestCase(unittest.TestCase):
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
    posts = lda.get_posts("localhost:27017", "steemdb_test")
    self.assertCountEqual([p for p in posts.columns], ["post_permlink", "author", "parent_permlink", "first_tag", "last_tag", "created", "topic", "topic_probability"])
    self.assertCountEqual(posts.loc[0].tolist(), ["@someauthor/link", "someauthor", "somelink", "tag1", "tag2", pd.Timestamp.now().round('60min'), 0, 0.9])

  def test_remove_usernames(self):
    post = lda.remove_usernames("some @guy said")
    self.assertEqual(post, "some said")

  def test_remove_html_tags(self):
    post = lda.remove_html_tags("<tag>Test</tag>")
    self.assertEqual(post, "Test")

  def test_unescape_html_tags(self):
    post = lda.unescape_html_tags("&lt;tag&gt;Test&lt;/tag&gt;")
    self.assertEqual(post, "<tag>Test</tag>")

  def test_convert_markdown_to_html(self):
    post = lda.convert_markdown("**Test**")
    self.assertEqual(post, "<p><strong>Test</strong></p>")

  def test_split_post(self):
    post = lda.split_post("продакшен - это зло, ресерч - офигенно.")
    self.assertCountEqual(post, ["продакшен", "это", "зло", "ресерч", "офигенно"])

  def test_remove_stopwords(self):
    post = lda.remove_stopwords(["я", "провожу", "тест"])
    self.assertCountEqual(post, ["провожу", "тест"])

  def test_lemmatize(self):
    post = lda.lemmatize(["привожу", "к", "начальной", "форме"])
    self.assertCountEqual(post, ["приводить", "к", "начальный", "форма"])

  def test_prepare_post(self):
    post = "@author Креативное &lt;пугало&gt; <html>\n<h1>**Пугало**</h1>\n![image.png](https://steemitimages.com/image.png)\n</html>"
    prepared_post = lda.prepare_post(post)
    self.assertCountEqual(prepared_post, ["креативный", "пугать", "пугать"])

  def test_remove_short_words(self):
    texts = [["some", "of", "these", "words", "are", "too", "short"], ["yea", "really"]]
    texts = lda.remove_short_words(texts)
    self.assertCountEqual(texts[0], ["some", "these", "words", "are", "too", "short"])

  def test_remove_short_texts(self):
    texts = [["some", "of", "these", "texts", "are", "too", "short"], ["yea", "really"]]
    texts = lda.remove_short_texts(texts)
    self.assertEqual(len(texts), 1)

  def test_remove_high_frequent_words(self):
    texts = [["a", "a", "a", "a", "b", "c", "d"], ["a", "d", "d", "d", "b"]]
    texts = lda.remove_high_frequent_words(texts)
    self.assertCountEqual(texts[0], ["b", "c", "d"])

  def test_remove_low_frequent_words(self):
    texts = [["a", "a", "a", "a", "b", "c", "d"], ["a", "d", "d", "d", "b"]]
    texts = lda.remove_low_frequent_words(texts)
    self.assertCountEqual(texts[0], ["a", "a", "a", "a", "d"])

  def test_create_dictionary(self):
    texts = [["a"], ["b"]]
    lda.create_dictionary(texts)
    assert corpora.Dictionary.load("golos-corpora.dict")

  def test_create_corpus(self):
    texts = [["a"], ["b"]]
    dictionary = corpora.Dictionary(texts)
    lda.create_corpus(texts, dictionary)
    assert corpora.MmCorpus('golos-corpora-bow.mm')

  