OKay, will this start it all?
# distributed-hyperopt
Distributed Hyperopt example on cifar10 

https://github.com/hyperopt/hyperopt

In order to run it:

1. Setup a Mongodb server
2. Specify os.env variables TF_CONFIG with PS_HOSTS and run .run.sh

# install mongo and open up port
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv EA312927
    echo "deb http://repo.mongodb.org/apt/ubuntu xenial/mongodb-org/3.2 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-3.2.list
    apt-get update
    apt-get install -y mongodb-org
    mongod --dbpath . --port $MONGO_DB_PORT --directoryperdb --fork --journal --logpath log.log --nohttpinterface

    echo "[+] Im a PS running Mongo"


On each worker node run:

    export MONGO_DB_HOST=<mongodb server address>
    export MONGO_DB_PORT=5000
    export EXPERIMENT_NAME=paperspace-hyperopt 
    ./run.sh
