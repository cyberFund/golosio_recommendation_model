# Golos.io recommendation system

This repo contains files of recommendation system for golos.io

```
.
+-- similar.ann - Saved ANN model for finding similar posts
+-- golos-corpora.dict - Saved dictionary for LDA
+-- golos-corpora_tfidf.mm* - Saved mm index
+-- golos.lda_model* - Saved LDA model
+-- mappings.pkl - Saved mappings for FFM model
+-- model.bin - Saved FFM model
+-- server.py - Flask server for recommendation system
+-- model
   +-- ann.py - Process of finding similar posts
   +-- lda.py - Process of finding LDA topics for each post
   +-- train.py - Process of finding LDA topics for each 
   +-- predict.py - Process of creating predictions
```
# Installation

To add tasks to cron tab:
```bash
$ install.sh DATABASE_HOST:DATABASE_PORT DATABASE_NAME
```

For example:
```bash
$ install.sh localhost:27017 steemdb_1
```

To remove tasks from cron tab, run:
```bash
$ uninstall.sh
```
# How to use it

To start server, run:
```bash
$ run.sh DATABASE_HOST:DATABASE_PORT DATABASE_NAME
```

For example:
```bash
$ run.sh localhost:27017 steemdb_1
```

To get supported user ids, run
```bash
$ curl localhost:8080/users
```

To get history for some user, run:
```bash
$ curl localhost:8080/history?user=USER_ID
```

For example:
```bash
$ curl localhost:8080/history?user=58158

[
  "@vik/test-redaktora-dlya-botov-ot-vik-11-10", 
  "@vox-populi/otchyot-kuratora-30-oktyabrya-5-noyabrya", 
  "@vp-freelance/4kpmi-rezultaty-ezhenedelnogo-konkursa-luchshaya-rabota-po-itogam-nedeli", 
  "@vp-freelance/khudozhestvennyi-perevod", 
  "@vp-freelance/konkursnaya-rabota-16-odnazhdy-na-rabote-ikra-belugi", 
  "@vp-freelance/realnosti-frilansa-mysli", 
  "@vp-freelance/rezultaty-konkursa-odnazhdy-na-rabote-za-oktyabr-2017-goda", 
  "@vp-freelance/treiding-kak-vid-frilansa"
]
```

To get similar posts for specified one, run:
```bash
$ curl localhost:8080/similar?permlink=POST_PERMLINK
```

For example:

```bash
$ curl localhost:8080/similar?permlink=@mrosenquist/time-to-build-a-new-pc

[
  "@xaliq/v-moskve-moshenniki-ukrali-15-mln-rublei-nalichnymi-pri-obmene-na-bitkoiny", 
  "@cryptojournal/kriptofond-nextblock-global-ne-budet-vykhodit-na-ipo-iz-za-predostavleniya-lozhnykh-dannykh", 
  "@hoanhduc/toi-deo-biet-viet-j-ca-nhe", 
  "@gryph0n/podarochnyi-byteball", 
  "@gayush07/bitdegree-uvlekatelnaya-obrazovatelnaya-platforma"
]
```

To get recommendations for specified user, run:
```bash
curl localhost:8080/recommendations?user=USER_ID
```

For example:
```bash
$ curl localhost:8080/recommendations?user=58158

[
  {
    "post_permlink": "@yukhimchuk/10-tipov-turistov-kotorye-ezdyat-na-ekskursii", 
    "prediction": 0.32709062099456787
  }, 
  {
    "post_permlink": "@vp-painting/pol-gogen-vyzov-voskresnomu-khudozhniku", 
    "prediction": 0.11999647319316864
  }, 
  {
    "post_permlink": "@bogdych/o-proekte-golos", 
    "prediction": 0.0844191461801529
  }, 
  {
    "post_permlink": "@vp-actionlife/gde-zvezdy-kak-blyudca-i-belye-ovcy-dorozhe", 
    "prediction": 0.06712445616722107
  }, 
  {
    "post_permlink": "@vp-freelance/kak-frilanseru-pravilno-rabotat-s-bazoi-klientov", 
    "prediction": 0.02885333076119423
  }, 
  {
    "post_permlink": "@vp-golos-radio/pochta-radio-sila-golosa-poeziya-olgi-silaevoi-sinilga", 
    "prediction": 0.02669009566307068
  }, 
  {
    "post_permlink": "@vp-liganovi4kov/6iqonm-top-5-khoroshikh-postov-ot-avtorov-novichkov-golosa", 
    "prediction": 0.016573332250118256
  }
]
```
