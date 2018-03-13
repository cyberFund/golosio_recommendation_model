python3 -m nltk.downloader all

(crontab -l ; echo "0 0 * * * doc2vec_train start") | crontab  -
(crontab -l ; echo "0 5 * * * ann_train start") | crontab  -
(crontab -l ; echo "0 10 * * * ffm_train start") | crontab  -

sync_comments start
recommendations_server start

