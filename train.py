import argparse
import logging
import os
import pickle as pkl
from typing import Dict, List, Tuple
import numpy as np
import xgboost as xgb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tree_method", type=str, default="auto")
    parser.add_argument("--predictor", type=str, default="auto")
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--reg_alpha", type=float, default=0)
    parser.add_argument("--reg_lambda", type=float, default=1)
    parser.add_argument("--gamma", type=float, default=0),
    parser.add_argument("--min_child_weight", type=float, default=1)
    parser.add_argument("--subsample", type=float, default=1)
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--objective", type=str)
    parser.add_argument("--num_class", type=int)
    parser.add_argument("--hite_rate_k", type=int, default=5)
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    # see https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md
    parser.add_argument("--output_data_dir", type=str,
                        default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model_dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str,
                        default=os.environ.get("SM_CHANNEL_VALIDATION"))

    args, _ = parser.parse_known_args()

    dtrain = xgb.DMatrix(args.train)
    dval = xgb.DMatrix(args.validation)

    watchlist = (
        [(dtrain, "train"), (dval, "validation")]
    )

    K = args.hite_rate_k

    def hit_rate_at_k(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
        ''' Hit rate at K. Ratio of predictions where the correct label is in the top K most probable classes.
        '''
        y = dtrain.get_label()
        y_in_top_k_classes = predt[np.isin(y, predt.argsort(axis=1)[:, -K:])]
        return f'Hit_rate_at_{K}', float(len(y_in_top_k_classes) / len(y))

    xgboost_train_params = {
        "disable_default_eval_metric": 1,
        "tree_method": args.tree_method,
        "predictor": args.predictor,
        "max_depth": args.max_depth,
        "eta": args.eta,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "verbosity": args.verbosity,
        "objective": args.objective,
        "num_class": args.num_class,
    }

    results: Dict[str, Dict[str, List[float]]] = {}
    model = xgb.train(
        params=xgboost_train_params,
        dtrain=dtrain,
        evals=watchlist,
        num_boost_round=args.num_round,
        feval=hit_rate_at_k,
        evals_result=results
    )

    model_location = args.model_dir + "/xgboost-model"
    pkl.dump(model, open(model_location, "wb"))
    logging.info("Stored trained model at {}".format(model_location))


# Make the trained model deployable for inference calls by defining the function
# "model_fn" which loads and returns the model.
# https://sagemaker.readthedocs.io/en/stable/frameworks/xgboost/using_xgboost.html#deploy-open-source-xgboost-models
def model_fn(model_dir):
    with open(os.path.join(model_dir, "xgboost-model"), "rb") as f:
        booster = pkl.load(f)
    return booster
