import pandas as pd 
from pymongo import MongoClient
import sys

if (__name__ == "__main__"):
  events = pd.read_csv(sys.argv[3], names=['user_id', 'event_type', 'value', 'refurl', 'created_at'])
  events["created_at"] = pd.to_datetime(events["created_at"])
  client = MongoClient(sys.argv[1])
  db = client[sys.argv[2]]
  db.raw_event.insert_many(events.to_dict('records'))