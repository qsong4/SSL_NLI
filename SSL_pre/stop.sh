ps aux | grep "pre_train.py" | grep -v grep | awk '{print $2}' | xargs kill -9