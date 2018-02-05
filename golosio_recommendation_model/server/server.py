from flask import Flask, jsonify, request, render_template
from config import config
import json
from flask_cors import CORS
import sys
from pymongo import MongoClient
import pandas as pd
import pdb
from model.train import get_events

database_url = sys.argv[1]
database_name = sys.argv[2]
events = get_events(database_url, database_name)

app = Flask(__name__)
CORS(app)

@app.route('/recommendations')
def recommendations():
  user = int(request.args.get("user"))
  client = MongoClient(database_url)
  db = client[database_name]
  recommendations_df = pd.DataFrame(list(db.recommendation.find(
    {
      'user_id': user
    }, {
      'post_permlink': 1,
      'prediction': 1
    }
  )))
  if (recommendations_df.shape[0] > 0):
    recommendations_df = recommendations_df.sort_values(["prediction"], ascending=[0])
    recommendations_json = recommendations_df.drop(["_id"], axis=1).to_dict('records')
    return jsonify(recommendations_json)
  else:
    return jsonify([])

@app.route('/users')
def users():
  return jsonify(events["user_id"].unique().tolist())

@app.route('/history')
def history():
  user = int(request.args.get("user"))
  user_events = events[(events["user_id"] == user) & (events["like"] > 0.7)]
  return jsonify(user_events["post_permlink"].unique().tolist())

@app.route('/similar')
def similar():
  permlink = request.args.get("permlink")
  client = MongoClient(database_url)
  db = client[database_name]
  comment = db.comment.find_one(
    {
      '_id': permlink[1:]
    }, {
      'similar_posts': 1,
      'similar_distances': 1
    }
  )
  if comment:
    return jsonify(list(zip(comment["similar_posts"], comment["similar_distances"])))
  else:
    return jsonify([])

if __name__ == '__main__':
  config(app)
  port = 8080 # Use desired port
  app.run(port=port)
