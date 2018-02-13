from flask import Flask, jsonify, request, render_template
from .config import config as config_flask
import json
from flask_cors import CORS
import sys
from pymongo import MongoClient
import pandas as pd
import pdb
from golosio_recommendation_model.model.utils import get_events
from golosio_recommendation_model.config import config

database_url = config['database_url']
database_name = config['database_name']
events = get_events(database_url, database_name)

app = Flask(__name__)
port = 8080 # Use desired port

@app.route('/users')
def users():
  return jsonify(events[events["like"] > 0.7]["user_id"].unique().tolist())

@app.route('/history')
def history():
  user = request.args.get("user")
  user_events = events[(events["user_id"] == user) & (events["like"] > 0.7)]
  return jsonify(user_events["post_permlink"].unique().tolist())

@app.route('/user_id')
def user_id():
  client = MongoClient(database_url)
  db = client[database_name]
  user_name = request.args.get("user_name")
  user = db.account.find_one(
    {
      'name': user_name
    }, {
      'user_id': 1
    }
  )
  return jsonify({'user_id': user['user_id']})

@app.route('/recommendations')
def recommendations():
  user = request.args.get("user")
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

@app.route('/similar')
def similar():
  permlink = request.args.get("permlink")
  client = MongoClient(database_url)
  db = client[database_name]
  comment = db.comment.find_one(
    {
      '_id': permlink[1:]
    }, {
      'committed_similar_posts': 1,
      'committed_similar_distances': 1
    }
  )
  if comment:
    return jsonify(list(zip(comment["committed_similar_posts"], comment["committed_similar_distances"])))
  else:
    return jsonify([])

@app.route('/post_recommendations')
def post_recommendations():
  permlink = request.args.get("permlink")
  user = request.args.get("user")
  client = MongoClient(database_url)
  db = client[database_name]
  comment = db.comment.find_one(
    {
      '_id': permlink[1:]
    }, {
      'committed_similar_posts': 1,
      'committed_similar_distances': 1
    }
  )
  recommendations_df = pd.DataFrame(list(db.recommendation.find(
    {
      'user_id': user,
      'post_permlink': {"$in": comment['committed_similar_posts']}
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

def run_recommendations_server():
  CORS(app)
  config_flask(app)
  app.run(port=port)

if __name__ == '__main__':
  run_server()
