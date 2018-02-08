from setuptools import setup

setup(
  name='golosio_recommendation_model',
  version='0.1',
  description='Recommendation system for golos.io',
  url='https://github.com/cyberFund/golosio-recommendation-model',
  author='',
  author_email='',
  license='MIT',
  packages=['golosio_recommendation_model'],
  zip_safe=False,
  scripts=['bin/ann_train', 'bin/doc2vec_train', 'bin/ffm_train',
          'bin/doc2vec_predict', 'bin/ann_predict', 'bin/ffm_predict',
          'bin/recommendations_server', 'bin/sync_comments', 'bin/sync_events']
)