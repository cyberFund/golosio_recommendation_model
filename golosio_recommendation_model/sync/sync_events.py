import pandas as pd 
from pymongo import MongoClient
import sys
from golosio_recommendation_model.model import utils
from golosio_recommendation_model.config import config
import pymysql

@utils.error_log("Sync events")
def get_events(events_host, events_database, events_user, events_password):
  result = []
  connection = pymysql.connect(host=events_host,
                             user=events_user,
                             password=events_password,
                             db=events_database,
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
  try:
    with connection.cursor() as cursor:
      sql = """
SELECT user_id, event_type, value, refurl, created_at
FROM golos.web_events 
WHERE 
   (event_type = "Comment" OR event_type = "Vote" OR event_type = "PageView") 
   AND created_at < CURDATE()
   AND created_at >  CURDATE() - INTERVAL 1 DAY
"""
      cursor.execute(sql)
      result = cursor.fetchall()
  finally:
    print(result)
    connection.close()
  return result
    
def sync_events():
  url = config['database_url']
  database = config['database_name']
  events_database = config['events_database']
  utils.log("Sync events", "Get events from a database...")
  events = get_events(
    events_database['host'], 
    events_database['database'], 
    events_database['user'], 
    events_database['password']
  )
  client = MongoClient(url)
  db = client[database]
  if len(events):
    db.raw_event.insert_many(events)
