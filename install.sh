(crontab -l ; echo "0 */8 * * * doc2vec_train start") | crontab  -
(crontab -l ; echo "0 */3 * * * ann_train start") | crontab  -
(crontab -l ; echo "0 */5 * * * ffm_train start") | crontab  -

sync_comments start
