from flask import Flask, jsonify, request, render_template
from config import config
import json
from flask_cors import CORS
import sys
from pymongo import MongoClient
import pandas as pd
import pdb

database_url = sys.argv[1]
database_name = sys.argv[2]

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

if __name__ == '__main__':
  config(app)
  # TODO add certificate
  app.run(port=8080, ssl_context='adhoc')
