#!/usr/bin/env bash

apt-get update -y
apt-get install -y netcat

echo "Copying objective function to hyerpopt worker dir"
MONGO_WORKER_PATH=$(dirname $(which hyperopt-mongo-worker))
cp ./cifar10_tutorial.py $MONGO_WORKER_PATH

echo $MONGO_DB_HOST $MONGO_DB_PORT

while :; do
    if nc -z -v $MONGO_DB_HOST $MONGO_DB_PORT 2>/dev/null; then
        echo "mongodb is up!"
        hyperopt-mongo-worker --mongo=$MONGO_DB_HOST:$MONGO_DB_PORT/foo_db --poll-interval=1.0 --exp-key=distributed-hyperopt
 &
        python ./tune_cifar10.py
        exit
    else
        nc -z -v $MONGO_DB_HOST $MONGO_DB_PORT
        echo "mongodb is down!"
        sleep 5
    fi
done



