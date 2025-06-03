import pandas as pd
import os
import time
import copy
from sklearn.preprocessing import MinMaxScaler

from helper import set_all_seeds
from constants import (
    PATH_TO_DATA,
    sm_bin_vars,
    sm_cont_vars,
    sm_structure,
    sm_monotone_constraints,
    lpmc_bin_vars,
    lpmc_cont_vars,
    lpmc_structure,
    lpmc_mono_cons,
)
from utils import augment_dataset, transform_vars_list, transform_mono_cons
from models_wrapper import RUMBoost, APLR
from rumboost.datasets import load_preprocess_LPMC, load_preprocess_SwissMetro
from rumboost.metrics import cross_entropy

dataset_loader = {
    "LPMC": load_preprocess_LPMC,
    "SwissMetro": load_preprocess_SwissMetro,
}

dataset_vars = {
    "LPMC": (lpmc_bin_vars, lpmc_cont_vars),
    "SwissMetro": (sm_bin_vars, sm_cont_vars),
}

dataset_structure = {
    "LPMC": lpmc_structure,
    "SwissMetro": sm_structure,
}
dataset_monotone_constraints = {
    "LPMC": lpmc_mono_cons,
    "SwissMetro": sm_monotone_constraints,
}
dataset_num_classes = {
    "LPMC": 4,
    "SwissMetro": 3,
}


def train(args):
    """
    Train the specified model.
    """

    if not args.outpath:
        args.outpath = f"results/{args.dataset}/{args.model}/{args.model_type}/mono{args.monotone}/"

    # create the output directory if it does not exist
    os.makedirs(args.outpath, exist_ok=True)

    # set the random seed for reproducibility
    set_all_seeds(args.seed)

    data_train, data_test, folds = dataset_loader[args.dataset](path=PATH_TO_DATA)

    target = "choice"
    y_train = data_train[target]
    y_test = data_test[target]

    bin_vars, cont_vars = dataset_vars[args.dataset]

    X_train = augment_dataset(data_train, cont_vars, type=args.model_type)
    X_test = augment_dataset(data_test, cont_vars, type=args.model_type)
    X_train_bin = augment_dataset(
        data_train, bin_vars, type="constant"
    )
    X_test_bin = augment_dataset(
        data_test, bin_vars, type="constant"
    )
    X_train = pd.concat([X_train, X_train_bin], axis=1)
    X_test = pd.concat([X_test, X_test_bin], axis=1)

    # transform the monotone constraints to a specific format
    monotone_constraints = transform_mono_cons(
        dataset_monotone_constraints[args.dataset],
        args.model_type,
        cont_vars,
    )
    if not args.monotone:
        monotone_constraints = {k: 0 for k in monotone_constraints.keys()}

    #binary variables are never boosted from parameter space
    new_bin_vars = transform_vars_list(
        bin_vars, "constant", bin_vars
    )
    bin_vars = new_bin_vars

    structure = copy.deepcopy(dataset_structure[args.dataset])
    for u in structure:
        structure[u] = transform_vars_list(
            structure[u], args.model_type, cont_vars
        ) + bin_vars
    new_cont_vars = transform_vars_list(
        cont_vars, args.model_type, cont_vars
    )
    cont_vars = new_cont_vars

    num_classes = dataset_num_classes[args.dataset]

    # scale the features
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    if args.model == "RUMBoost":
        model = RUMBoost(
            structure=structure,
            monotone_constraints=monotone_constraints,
            cont_vars=cont_vars,
            bin_vars=bin_vars,
            num_classes=num_classes,
            args=args,
        )
        save_path = (
            args.outpath + f"model.json"
        )
    elif args.model == "APLR":
        model = APLR(
            monotone_constraints=monotone_constraints,
            args=args,
        )
        save_path = (
            args.outpath + f"model.json"
        )

    model.build_dataloader(X_train_scaled, y_train, folds=folds)

    # fit the model
    start_time = time.time()
    best_train_loss, num_iterations = model.fit()
    end_time = time.time()

    # test the model
    preds = model.predict(X_test_scaled)

    loss_test = cross_entropy(preds, y_test)

    print(f"Best Train Loss: {best_train_loss}, Best Test Loss: {loss_test}")

    results_dict = {
        "train_loss": best_train_loss,
        "train_time": end_time - start_time,
        "loss_test": loss_test,
        "num_iterations": num_iterations,
    }

    # save the results
    pd.DataFrame(results_dict, index=[0]).to_csv(
        args.outpath + f"results.csv"
    )

    if args.save_model:
        model.save_model(save_path)
