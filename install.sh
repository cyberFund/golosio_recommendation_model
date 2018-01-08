(crontab -l ; echo "0 5 * * * python3 `pwd`/model/lda.py") | crontab  -
(crontab -l ; echo "0 5 * * * python3 `pwd`/model/ann.py") | crontab  -
(crontab -l ; echo "0 5 * * * python3 `pwd`/model/train.py") | crontab  -
(crontab -l ; echo "0 5 * * * python3 `pwd`/model/predict.py") | crontab  -
