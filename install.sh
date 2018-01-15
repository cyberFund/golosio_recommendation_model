(crontab -l ; echo "0 */5 * * * cd `pwd` && python3 ./model/lda.py $1 $2") | crontab  -
(crontab -l ; echo "0 */5 * * * cd `pwd` && python3 ./model/doc2vec.py $1 $2") | crontab  -
(crontab -l ; echo "0 */5 * * * cd `pwd` && python3 ./model/ann.py $1 $2") | crontab  -
(crontab -l ; echo "0 */5 * * * cd `pwd` && python3 ./model/train.py $1 $2") | crontab  -
(crontab -l ; echo "0 */5 * * * cd `pwd` && python3 ./model/predict.py $1 $2") | crontab  -
