(crontab -l ; echo "0 */10 * * * cd `pwd` && python3 ./model/doc2vec.py $1 $2") | crontab  -
(crontab -l ; echo "30 */5 * * * cd `pwd` && python3 ./model/ann.py $1 $2") | crontab  -
(crontab -l ; echo "0 */10 * * * cd `pwd` && python3 ./model/train.py $1 $2") | crontab  -
(crontab -l ; echo "30 */5 * * * cd `pwd` && python3 ./model/predict.py $1 $2") | crontab  -
