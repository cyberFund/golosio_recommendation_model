(crontab -l ; echo "0 */10 * * * cd `pwd` && python3 ./model/train/doc2vec.py $1 $2") | crontab  -
(crontab -l ; echo "0 */5 * * * cd `pwd` && python3 ./model/train/ann.py $1 $2") | crontab  -
(crontab -l ; echo "0 */10 * * * cd `pwd` && python3 ./model/train/ffm.py $1 $2") | crontab  -

(crontab -l ; echo "0 */1 * * * cd `pwd` && python3 ./model/predict/doc2vec.py $1 $2") | crontab  -
(crontab -l ; echo "0 */1 * * * cd `pwd` && python3 ./model/predict/ann.py $1 $2") | crontab  -
(crontab -l ; echo "0 */5 * * * cd `pwd` && python3 ./model/predict/ffm.py $1 $2") | crontab  -