import os
import json
from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials
from pymongo.errors import ServerSelectionTimeoutError

from .cifar10_tutorial import train_cifar10


def main():
    mongo_db_host = os.environ["MONGO_DB_HOST"]
    mongo_db_port = os.environ["MONGO_DB_PORT"]
    experiment_name = os.environ["EXPERIMENT_NAME"]

    mongo_connect_str = "mongo://{0}:{1}/foo_db/jobs".format(mongo_db_host, mongo_db_port)

    while True:
        try:
            trials = MongoTrials(mongo_connect_str, exp_key=experiment_name)
        except ServerSelectionTimeoutError:
            pass
        else:
            space = {
                'lr': hp.loguniform('lr', -10, 2),
                'momentum': hp.uniform('momentum', 0.1, 0.9)
            }
            best = fmin(train_cifar10, space=space, trials=trials, algo=tpe.suggest, max_evals=25)

            if os.environ["JOB_NAME"] == "ps":
                save_path = os.path.join("./logs", "results.json")
                with open(save_path, "w") as f:
                    json.dump(json.dumps(best), f)

            return


if __name__ == "__main__":
    main()
