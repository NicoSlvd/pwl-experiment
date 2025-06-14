import argparse

def parse_cmdline_args(raw_args=None, parser=None):

    if parser is None:
        parser = argparse.ArgumentParser()
        
    parser.add_argument(
        "--outpath",
        type=str,
        default="",
        help="Output path for the model",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="constants",
        help="model type for rumboost",
        choices=["constant", "linear", "cubic", "constant_linear", "constant_cubic"],
    )

    parser.add_argument(
        "--optim_interval",
        type=int,
        default=20,
        help="Optimisation interval for the ordinal model",
    )

    parser.add_argument(
        "--num_leaves",
        type=int,
        default=31,
        help="Number of leaves in the tree",
    )
    parser.add_argument(
        "--min_gain_to_split",
        type=float,
        default=0.0,
        help="Minimum gain to split the tree",
    )
    parser.add_argument(
        "--min_sum_hessian_in_leaf",
        type=float,
        default=0.001,
        help="Minimum sum hessian in leaf",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate for the model",
    )
    parser.add_argument(
        "--max_bin",
        type=int,
        default=64,
        help="Maximum number of bins for the model",
    )
    parser.add_argument(
        "--min_data_in_bin",
        type=int,
        default=3,
        help="Minimum number of data in bin",
    )
    parser.add_argument(
        "--min_data_in_leaf",
        type=int,
        default=20,
        help="Minimum number of data in leaf",
    )
    parser.add_argument(
        "--feature_fraction",
        type=float,
        default=1.,
        help="Feature fraction for the model",
    )
    parser.add_argument(
        "--bagging_fraction",
        type=float,
        default=1.,
        help="Bagging fraction for the model",
    )
    parser.add_argument(
        "--bagging_freq",
        type=int,
        default=0,
        help="Bagging frequency for the model",
    )
    parser.add_argument(
        "--lambda_l1",
        type=float,
        default=0.0,
        help="L1 regularisation for the model",
    )
    parser.add_argument(
        "--lambda_l2",
        type=float,
        default=0.0,
        help="L2 regularisation for the model",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=6000,
        help="Number of iterations for the model",
    )
    parser.add_argument(
        "--early_stopping_rounds",
        type=int,
        default=100,
        help="Early stopping rounds for the model",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of threads for the model",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity of the model",
    )
    parser.add_argument(
        "--verbose_interval",
        type=int,
        default=10,
        help="Verbosity interval of the model evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the model",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.64,
        help="Train size for the model",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.16,
        help="Validation size for the model",
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default="false",
        help="Save the model to disk",
        choices=["true", "false"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RUMBoost",
        help="Model to train",
        choices=["RUMBoost", "APLR"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="LPMC",
        help="Dataset to use for training",
        choices=["LPMC", "SwissMetro"],
    )
    parser.add_argument(
        "--monotone",
        type=str,
        default="false",
        help="Use monotone constraints for the model",
        choices=["true", "false"],
    )
    parser.add_argument(
        "--all_boosters",
        type=str,
        default="false",
        help="if using all boosters to update at each boosting iteration",
        choices=["true", "false"],
    )

    parser.set_defaults(feature=True)
    args = parser.parse_args(raw_args)

    d = {'true': True,
         'false': False}


    args.save_model = d[args.save_model]
    args.monotone = d[args.monotone]
    args.all_boosters = d[args.all_boosters]

    return args
