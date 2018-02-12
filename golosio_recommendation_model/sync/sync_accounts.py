import pandas as pd 
from pymongo import MongoClient
import sys
from golosio_recommendation_model.model import utils
from golosio_recommendation_model.config import config

@utils.error_log("Sync accounts")
def sync_accounts():
  url = config['database_url']
  database = config['database_name']
  accounts_path = config['accounts_path']
  utils.log("Sync accounts", "Get accounts from a file...")
  accounts = pd.read_csv(accounts_path, names=[])
  client = MongoClient(url)
  db = client[database]
  db.account.drop()
  db.account.insert_many(events.to_dict('records'))