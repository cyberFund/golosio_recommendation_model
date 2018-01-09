# Golos.io recommendation system

This repo contains files of recommendation system for golos.io

golosio-recommendation-model
|- similar.ann - Saved ANN model for finding similar posts
|- golos-corpora.dict - Saved dictionary for LDA
|- golos-corpora_tfidf.mm* - Saved mm index
|- golos.lda_model* - Saved LDA model
|- mappings.pkl - Saved mappings for FFM model
|- model.bin - Saved FFM model
|- server.py - Flask server for recommendation system
|- model
   |- ann.py - Process of finding similar posts
   |- lda.py - Process of finding LDA topics for each post
   |- train.py - Process of finding LDA topics for each 
   |- predict.py - Process of creating predictions

# Installation

To add tasks to cron tab:
```bash
install.sh HOST:PORT DATABASE
```

To remove tasks from cron tab, run:
```bash
uninstall.sh
```
# How to use it

To start server, run:
```bash
run.sh HOST:PORT DATABASE
```

To get supported user ids, run
```bash
curl localhost:8080/users
```

To get history for some user, run:
```bash
curl localhost:8080/history?user=USER_ID
```

To get similar posts for specified one, run:
```bash
curl localhost:8080/similar?permlink=POST_PERMLINK
```

To get recommendations for specified user, run:
```bash
curl localhost:8080/recommendations?user=USER_ID
```