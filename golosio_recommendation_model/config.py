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
