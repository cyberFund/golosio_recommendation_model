from flask import Flask, jsonify, request, render_template
from config import config
import json
from flask_cors import CORS
import sys
from pymongo import MongoClient
import pandas as pd
import pdb

events = pd.read_csv(sys.argv[1])
database_url = sys.argv[2]
database_name = sys.argv[3]

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
  recommendations_json = recommendations_df.drop(["_id"], axis=1).to_dict('records')
  return jsonify(recommendations_json)

@app.route('/users')
def users():
  return jsonify(events["user_id"].unique().tolist())

@app.route('/history')
def history():
  user = int(request.args.get("user"))
  user_events = events[(events["user_id"] == user) & (events["like"] >= 0.7)]
  return jsonify(user_events["post_permlink"].unique().tolist())

if __name__ == '__main__':
  config(app)
  # TODO add certificate
  app.run(port=8080, ssl_context='adhoc')
