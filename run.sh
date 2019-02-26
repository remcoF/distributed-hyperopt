#!/usr/bin/env bash

#export MONGO_DB_HOST==$(echo $PS_HOSTS | awk -F ':' '{print $1}')
#export MONGO_DB_PORT=27017
#export EXPERIMENT_NAME="howdy"

apt-get update -y
apt-get install -y netcat

echo "Copying objective function to hyerpopt worker dir"
MONGO_WORKER_PATH=$(dirname $(which hyperopt-mongo-worker))
cp ./cifar10_tutorial.py $MONGO_WORKER_PATH

if [[ $JOB_NAME == "ps" ]]; then
    # install mongo and open up port
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv EA312927
    echo "deb http://repo.mongodb.org/apt/ubuntu xenial/mongodb-org/3.2 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-3.2.list
    apt-get update
    apt-get install -y mongodb-org
    mongod --dbpath . --port $MONGO_DB_PORT --directoryperdb --fork --journal --logpath log.log --nohttpinterface

    echo "[+] Im a PS running Mongo"
else
    echo "[+] Im a worker ready for action..."
fi

while :; do
    if nc -z $MONGO_DB_HOST $MONGO_DB_PORT 2>/dev/null; then
        echo "mongodb is up!"
        hyperopt-mongo-worker --mongo=$MONGO_DB_HOST:$MONGO_DB_PORT/foo_db --poll-interval=1.0 --exp-key=$EXPERIMENT_NAME &
        python ./tune_cifar10.py
        exit
    else
        echo "mongodb is down!"
        sleep 5
    fi
done



