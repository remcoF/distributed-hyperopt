import math
import os
import json
import math
import logging
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
            logging.debug('Launching MongoTrials for {}'.format(experiment_name))
            trials = MongoTrials(mongo_connect_str, exp_key=experiment_name)
        except ServerSelectionTimeoutError:
            logging.debug('No MongoDB server is available for an operation')
            pass
        else:
            space = {
                'x': hp.uniform('x', -2, 2)
            }
            best = fmin(obj, space=space, trials=trials, algo=tpe.suggest, max_evals=100)

            if os.environ["TYPE"] == "ps":
                save_path = os.path.join(data_dir, "results.json")
                with open(save_path, "w") as f:
                    logging.debug('Saving results.json to {}'.format(data_dir))
                    logging.info('Results: {}'.format((str(best))))
                    json.dump(json.dumps(best), f)
            return


if __name__ == "__main__":
    main()
