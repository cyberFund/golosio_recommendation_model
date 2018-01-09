(crontab -l ; echo "0 */5 * * * python3 `pwd`/model/lda.py $2 $3") | crontab  -
(crontab -l ; echo "0 */5 * * * python3 `pwd`/model/ann.py $2 $3") | crontab  -
(crontab -l ; echo "0 */5 * * * python3 `pwd`/model/train.py $1 $2 $3") | crontab  -
(crontab -l ; echo "0 */5 * * * python3 `pwd`/model/predict.py $1 $2 $3") | crontab  -