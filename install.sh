(crontab -l ; echo "0 5 * * * python `pwd`/model/lda.py") | crontab  -
(crontab -l ; echo "0 5 * * * python `pwd`/model/ann.py") | crontab  -
(crontab -l ; echo "0 5 * * * python `pwd`/model/train.py") | crontab  -
(crontab -l ; echo "0 5 * * * python `pwd`/model/predict.py") | crontab  -