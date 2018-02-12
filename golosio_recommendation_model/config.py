config = {
  'database_url': "localhost:27017", # Your mongo database url
  'database_name': "golos_comments", # Mongo database with dumps content
  'events_path': "/home/anatoli/Documents/golosio_recommendation_model/test_events4.csv", # Path to csv file with events
  'accounts_path': "/home/anatoli/Documents/golosio_recommendation_model/test_accounts.csv", # Path to csv file with accounts
  'node_url': 'ws://localhost:8090', # Golos.io websocket url
  'model_path': "/tmp/", # Path to model files
  'log_path': "/tmp/recommendation_model.log", # Path to model log
}