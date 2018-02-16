@echo on

start activate gluon
python cnnEnsemble.py 0
python cnnEnsemble.py 1
python cnnEnsemble.py 2
python cnnEnsemble.py 3
python cnnEnsemble.py 4
exit