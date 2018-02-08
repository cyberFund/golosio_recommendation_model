import pandas as pd 
from pymongo import MongoClient
import sys
from golosio_recommendation_model.model import utils
from golosio_recommendation_model.config import config

@utils.error_log("Sync events")
def sync_events():
  url = config['database_url']
  database = config['database_name']
  events_path = config['events_path']
  utils.log("Sync events", "Get events from a file...")
  events = pd.read_csv(events_path, names=['user_id', 'event_type', 'value', 'refurl', 'created_at'])
  events["created_at"] = pd.to_datetime(events["created_at"])
  client = MongoClient(url)
  db = client[database]
  db.raw_event.insert_many(events.to_dict('records'))