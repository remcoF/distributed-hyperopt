import math
import os
import json
import math
from time import sleep
from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials
from pymongo.errors import ServerSelectionTimeoutError


def obj(params):
    x = params['x']
    sleep(3)
    return math.sin(x)


def main():
    mongo_db_host = os.environ["MONGO_DB_HOST"]
    mongo_db_port = os.environ["MONGO_DB_PORT"]
    experiment_name = os.environ["EXPERIMENT_NAME"]
    data_dir = os.path.abspath(os.environ.get('PS_MODEL_PATH', os.getcwd()))

    mongo_connect_str = "mongo://{0}:{1}/foo_db/jobs".format(mongo_db_host, mongo_db_port)

    while True:
        try:
            trials = MongoTrials(mongo_connect_str, exp_key=experiment_name)
        except ServerSelectionTimeoutError:
            pass
        else:
            space = {
                'x': hp.uniform('x', -2, 2)
            }
            best = fmin(obj, space=space, trials=trials, algo=tpe.suggest, max_evals=100)

            if os.environ["JOB_NAME"] == "ps":
                save_path = os.path.join(data_dir, "results.json")
                with open(save_path, "w") as f:
                    json.dump(json.dumps(best), f)
                    print(str(best))
            return


if __name__ == "__main__":
    main()
