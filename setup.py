from setuptools import setup

setup(
  name='golosio_recommendation_model',
  version='1.0.0',
  description='Recommendation system for golos.io',
  url='https://github.com/cyberFund/golosio_recommendation_model',
  author='',
  author_email='',
  license='MIT',
  packages=['golosio_recommendation_model', 'golosio_recommendation_model/model', 
            'golosio_recommendation_model/model/predict', 'golosio_recommendation_model/model/train', 
            'golosio_recommendation_model/server', 'golosio_recommendation_model/sync'],
  zip_safe=False,
  scripts=['bin/ann_train', 'bin/doc2vec_train', 'bin/ffm_train',
          'bin/doc2vec_predict', 'bin/ann_predict', 'bin/ffm_predict',
          'bin/recommendations_server', 'bin/sync_comments', 'bin/sync_events',
          'bin/sync_accounts']
)
