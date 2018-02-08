crontab  -l | grep -v 'doc2vec_train' | crontab  -
crontab  -l | grep -v 'ann_train'  | crontab  -
crontab  -l | grep -v 'ffm_train'  | crontab  -

doc2vec_train stop
ann_train stop
ffm_train stop

doc2vec_predict stop
ann_predict stop
ffm_predict stop

sync_comments stop
