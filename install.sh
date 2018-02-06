(crontab -l ; echo "0 */10 * * * doc2vec_train $1 $2") | crontab  -
(crontab -l ; echo "0 */5 * * * ann_train $1 $2") | crontab  -
(crontab -l ; echo "0 */10 * * * ffm_train $1 $2") | crontab  -

(crontab -l ; echo "0 */1 * * * doc2vec_predict $1 $2") | crontab  -
(crontab -l ; echo "0 */1 * * * ann_predict $1 $2") | crontab  -
(crontab -l ; echo "0 */5 * * * ffm_predict $1 $2") | crontab  -
