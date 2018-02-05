./gdrive download --recursive 1h1xEERnG7Z69KGSXLIjZj5e3ygh5oeL4
cd ./golosio-recommendation-model
mv * ../
cd ../
rm -rf ./golosio-recommendation-model

(crontab -l ; echo "0 */10 * * * cd `pwd` && python3 ./model/doc2vec.py $1 $2") | crontab  -
(crontab -l ; echo "5 */10 * * * cd `pwd` && python3 ./model/ann.py $1 $2") | crontab  -
(crontab -l ; echo "10 */10 * * * cd `pwd` && python3 ./model/train.py $1 $2") | crontab  -
(crontab -l ; echo "15 */10 * * * cd `pwd` && python3 ./model/predict.py $1 $2") | crontab  -