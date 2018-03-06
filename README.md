# Golos.io recommendation system

This repo contains files of recommendation system for golos.io

```
.
+-- install.sh - Bash script to fill crontab tasks for a model rebuilding
+-- uninstall.sh - Bash script to clean crontab and to stop all daemons
+-- setup.py - Package configuration
+-- golosio_recommendation_model
   +-- config.py - Overall model configuration
   +-- daemonize.py - Function for making daemon of a specified function
   +-- server
      +-- server.py - Flask server for recommendation system
      +-- config.py - Server configuration
   +-- sync
      +-- convert_events.py - Convert events in MongoDB for training FFM model
      +-- sync_comments.py - Synchronizing MongoDB with Golos node
      +-- sync_events.py - Synchronizing Golosio MySQL events with MongoDB
      +-- sync_accounts.py - Synchronizing Golosio MySQL accounts with MongoDB
   +-- model
      +-- utils.py - Helpers for preprocessing, processes regulation and etc.
      +-- train
         +-- ann.py - Process of training model to find similar posts
         +-- doc2vec.py - Process of training model to find doc2vec vectors for each post
         +-- ffm.py - Process of training FFM model to arrange recommendations for each user
      +-- predict
         +-- ann.py - Process of finding similar posts for new posts in database
         +-- doc2vec.py - Process of finding doc2vec vectors for each new post in database
         +-- ffm.py - Process of creating recommendations list for each active user
+-- bin - These scripts will appear in /usr/local/bin directory
   +-- doc2vec_train - Daemon that trains doc2vec model
   +-- doc2vec_predict - Daemon that makes doc2vec predictions for all posts in database
   +-- ann_train - Daemon that trains ANN model
   +-- ann_predict - Daemon that makes ANN predictions for all posts in database
   +-- ffm_train - Daemon that trains FFM model
   +-- ffm_predict - Daemon that makes FFM predictions and stores them to a database
   +-- recommendations_server - Daemon for a recommendation model server
   +-- sync_comments - Daemon that loads new comments from a golos node to a database
   +-- sync_events - Daemon that loads events from a specified csv file to a database
   +-- sync_accounts - Daemon that loads accounts from a specified csv file to a database
```

# Architecture

Recommendation model architecture: ![Recommendation model architecture](architecture.png)

# Installation

Install LibFFM before usage. Instruction can be found here: http://github.com/alexeygrigorev/libffm-python

Prepare mongo database before installation. You can load current mongo dumps here:
```bash
$ scp earth@earth.cyber.fund:~/Documents/golosio-recommendation-model/golosio-recommendation-dump-comment.json ./
$ scp earth@earth.cyber.fund:~/Documents/golosio-recommendation-model/golosio-recommendation-dump-event.json ./
```

To load events to a mongo database from mysql database later, use 
```bash
$ sync_events start
```

Prepare config file before installation. It should looks like this:
```python
# golosio_recommendation_model/config.py
config = {
  'database_url': "localhost:27017", # Your mongo database url
  'database_name': "golos_comments", # Mongo database with dumps content
  'accounts_path': "/home/anatoli/Documents/golosio_recommendation_model/accounts.csv", # Path to csv file with accounts, only for debug
  'node_url': 'ws://localhost:8090', # Golos.io websocket url
  'model_path': "/tmp/", # Path to model files
  'log_path': "/tmp/recommendation_model.log", # Path to model log
  'events_database': { # Credentials for mysql database with events
    'host': 'localhost',
    'database': 'golos',
    'user': 'root',
    'password': 'root'
  }
}
```

Install a package with:
```bash
$ pip3 install .
```

To add model daemons to a crontab, use:
```bash
$ install.sh
```
This script will add train tasks to a crontab and will start comments synchronization. 

It'll take some time to generate a new version of a model. For example, You'll get new model after a full day, if you ran installation script at 22:00. If you want to get first version as quickly as possible, run daemons manually:
```bash
$ doc2vec_train start
$ ann_train start
$ ffm_train start
```

To stop model daemons and to clean crontab, run:
```bash
$ uninstall.sh
```

# How to use it

To add new events to a database, run:
```bash
$ sync_events start
```

To update accounts in a database, run:
```bash
$ sync_accounts start
```

To start server, run:
```bash
$ recommendations_server start
```

## Base server actions

To get similar posts and distances to each of them for a specified one, run:
```bash
$ curl http://localhost:8080/similar?permlink=POST_PERMLINK
```

For example:

```bash
$ curl http://localhost:8080/similar?permlink=@cka3ka/0x-zrx-naverno-zatuzemunit-skoro-50-50

[
  [
    "@cka3ka/0x-zrx-naverno-zatuzemunit-skoro-50-50", 
    0.00017143443983513862
  ], 
  [
    "@cka3ka/golos-tuzemun", 
    0.11337035149335861
  ], 
  [
    "@cka3ka/3oo5vp-sozdatel-ethereum-vitalik-buterin-voshel-v-spisok-50", 
    0.528253972530365
  ], 
  [
    "@cka3ka/sozdatel-ethereum-vitalik-buterin-voshel-v-spisok-50", 
    0.6705115437507629
  ], 
  [
    "@cka3ka/bitcoin-stal-shestym-po-populyarnosti-sredi-mirovykh-valyut", 
    0.7635799050331116
  ], 
  [
    "@abdulazizov/v-zimbabe-bitkoin-likhoradka", 
    0.9337025880813599
  ], 
  [
    "@abdulazizov/bitcoin-fork", 
    0.937701404094696
  ],
  ...
]
```

To get recommendations for specified user, run:
```bash
curl http://localhost:8080/recommendations?user=USER_ID
```

For example:
```bash
$ curl http://localhost:8080/recommendations?user=58158

[
  {
    "post_permlink": "@tarimta/obektivnyi-marafon-etap-3", 
    "prediction": 0.9400154948234558
  }, 
  {
    "post_permlink": "@lumia/estafeta-prodolzhi-pesnyu-zadushevnaya", 
    "prediction": 0.9309653043746948
  }, 
  {
    "post_permlink": "@oksi969/dizain-cheloveka-lyubov-i-napravlenie-g-centr", 
    "prediction": 0.9016984701156616
  }, 
  {
    "post_permlink": "@is-pain/vzveshennye-lyudi-or-minus-16-kilogramm-za-dva-mesyaca", 
    "prediction": 0.8760964870452881
  }, 
  {
    "post_permlink": "@miroslav/golos-photography-awards-edinstvennaya", 
    "prediction": 0.8590876460075378
  },
  ...
]
```

To get recommendations for specified user and specified post, run:

```bash
curl http://localhost:8080/post_recommendations?user=USER_ID&permlink=POST_PERMLINK
```

For example:
```bash
$ curl http://localhost:8080/post_recommendations?user=71116&permlink=@cka3ka/golos-tuzemun

[
  {
    "post_permlink": "@cka3ka/0x-zrx-naverno-zatuzemunit-skoro-50-50", 
    "prediction": 0.34954845905303955
  }, 
  {
    "post_permlink": "@igrinov50-50/skonchalsya-leonid-bronevoi", 
    "prediction": 0.3138478994369507
  }, 
  {
    "post_permlink": "@ljpromo/isportilas-autentichnost", 
    "prediction": 0.16339488327503204
  }, 
  {
    "post_permlink": "@cka3ka/bitcoin-stal-shestym-po-populyarnosti-sredi-mirovykh-valyut", 
    "prediction": 0.07751113921403885
  }
]
```

## Debug server actions

To get supported user ids, run
```bash
$ curl http://localhost:8080/users
```

To find user id by username, run:
```bash
$ curl http://localhost:8080/user_id?user_name=USER_NAME
```

For example:
```bash
$ curl http://localhost:8080/user_id?user_name=smartcity-admin

{
  "user_id": 60837
}
```

To get page views for some user, run:
```bash
$ curl http://localhost:8080/history?user=USER_ID
```

For example:
```bash
$ curl http://localhost:8080/history?user=58158

[
  "@vik/test-redaktora-dlya-botov-ot-vik-11-10", 
  "@vox-populi/otchyot-kuratora-30-oktyabrya-5-noyabrya", 
  "@vp-freelance/4kpmi-rezultaty-ezhenedelnogo-konkursa-luchshaya-rabota-po-itogam-nedeli", 
  "@vp-freelance/khudozhestvennyi-perevod", 
  "@vp-freelance/konkursnaya-rabota-16-odnazhdy-na-rabote-ikra-belugi", 
  "@vp-freelance/realnosti-frilansa-mysli", 
  "@vp-freelance/rezultaty-konkursa-odnazhdy-na-rabote-za-oktyabr-2017-goda", 
  "@vp-freelance/treiding-kak-vid-frilansa"
]
```

# Configuration

Overall service configuration situated in config.py file, but most of the configuration hidden in .py files deep inside a package. You can additionally modify lines below to get better results: 

You can change service port here:

```python
# server/server.py
port = 8080 # Use desired port
```

It's highly recommended not to play with these parameters, but you can do it at your own risk in these files:
```python
# sync/convert_comments.py
...
HOURS_LIMIT = 14 * 24 # Time window size (in hours) for events extraction. Bigger values makes recommendations less sensitive to changes in preferences
...
# model/train/doc2vec.py
...
WORD_LENGTH_QUANTILE = 10 # Remove words shorter than 90%
TEXT_LENGTH_QUANTILE = 66 # Remove texts shorter than 66%
HIGH_WORD_FREQUENCY_QUANTILE = 99.5 # Remove words that appears more often than 99.5%
LOW_WORD_FREQUENCY_QUANTILE = 60 # Remove words that appears less often than 30%
# Parameters of doc2vec model. You can read about them in this article: https://radimrehurek.com/gensim/models/doc2vec.html
DOC2VEC_PARAMETERS = {
  'size': 300,
  'window': 20,
  'min_count': 5,
  'workers': 13
}
...
# model/train/ann.py
...
NUMBER_OF_TREES = 1000 # Number of trees in the ANN model. Bigger values ​​means slow prediction and high quality of result
NUMBER_OF_VALUES = 1000 # Number of values ​​for one-hot encoding of categorical features. Bigger values ​​means slow preparation and high quality of result
...
# model/train/ffm.py
...
# FFM parameters. You can read about them in this article: https://github.com/alexeygrigorev/libffm-python
MODEL_PARAMETERS = {
  'eta': 0.1,
  'lam': 0.01,
  'k': 70
}

ITERATIONS = 10 # Iterations of training process
WORKERS = 13 # Number of workers for dataset processing. Should be equal to AVAILABLE_CORES + 1
...
# model/predict/doc2vec.py
...
DOC2VEC_STEPS = 2500 # Number of steps for doc2vec model. Bigger values ​​means slow prediction and fast convergence
DOC2VEC_ALPHA = 0.03 # Learning rate for doc2vec model. Bigger values ​​means fast prediction and slow convergence
...
# model/predict/ann.py
...
NUMBER_OF_RECOMMENDATIONS = 50 # Number of similar posts for a post
...
```

# Tests and logs

To run load tests, download first version of a model and use:
```bash
python3 -m unittest tests.load_test_case
```

It'll show average response time for actions that returns recommendations and similar posts.

To see model logs, run:
```bash
tail -f ./model.log
```

# Timing
Processing time:
- doc2vec - 2.5h
    - train - 1.5h
    - predict - 1h
- ann - 2h
    - train - 1h
    - predict 1h
- ffm - 5h
    - train - 4h
    - predict - 1h

Tested on a server with i7-5930K, 128Gb DDR-4, 1 Tb SSD-PCIe.
