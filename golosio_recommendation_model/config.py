import os

config = {
  'database_url': os.environ.get('GOLOSIO_DATABASE_URL', "localhost:27017"), # Your mongo database url
  'database_name': os.environ.get('GOLOSIO_DATABASE_NAME', "golos_comments"), # Mongo database with dumps content
  'accounts_path': os.environ.get('GOLOSIO_ACCOUNTS_PATH', "/home/anatoli/Documents/golosio_recommendation_model/accounts.csv"), # Path to csv file with accounts, only for debug
  'node_url': os.environ.get('GOLOSIO_NODE_URL', 'http://localhost:8090'), # Golos.io API url
  'model_path': os.environ.get("GOLOSIO_MODEL_PATH", "/tmp/"), # Path to model files
  'log_path': os.environ.get("GOLOSIO_LOG_PATH", "/tmp/recommendation_model.log"), # Path to model log
  'events_database': { # Credentials for mysql database with events
    'host': os.environ.get('GOLOSIO_EVENTS_HOST', 'localhost'),
    'database': os.environ.get('GOLOSIO_EVENTS_DATABASE', 'golos'),
    'user': os.environ.get('GOLOSIO_EVENTS_USER', 'root'),
    'password': os.environ.get('GOLOSIO_EVENTS_PASSWORD', 'root')
  }
}
