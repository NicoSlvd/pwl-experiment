import pandas as pd
import lightgbm as lgb
import numpy as np
import copy

from utils import (
    generate_general_params,
    generate_rum_structure,
    generate_boost_from_param_space,
    add_hyperparameters,
    build_lgb_dataset,
)

from interpret.glassbox import APLRClassifier

from rumboost.rumboost import rum_train
from rumboost.rumboost import RUMBoost as load_rumboost
from rumboost.metrics import cross_entropy


class RUMBoost:
    """
    Wrapper class for RUMBoost model.
    """

    def __init__(self, **kwargs):
        # if alt_spec_features or socio_demo_chars are not None, generate rum structure
        self.rum_structure = generate_rum_structure(
            kwargs.get("structure"),
            kwargs.get("monotone_constraints"),
        )

        self.boost_from_parameter_space = generate_boost_from_param_space(
            self.rum_structure,
            kwargs.get("cont_vars"),
        )

        # generate general params
        general_params = generate_general_params(
            num_classes=kwargs.get("num_classes", 4),
            num_iterations=kwargs.get("args").num_iterations,
            early_stopping_rounds=kwargs.get("args").early_stopping_rounds,
            verbose=kwargs.get("args").verbose,
            verbose_interval=kwargs.get("args").verbose_interval,
        )

        num_boosters_per_util = np.min(
            [len(s) for s in kwargs.get("structure").values()]
        ) * kwargs.get("num_classes")

        general_params["max_booster_to_update"] = kwargs.get("num_classes") 
        general_params["boost_from_parameter_space"] = self.boost_from_parameter_space
        general_params["optim_interval"] = 0

        lr = kwargs.get("args").learning_rate

        # add hyperparameters
        hyperparameters = {
            "learning_rate": lr,
            "num_leaves": kwargs.get("args").num_leaves,
            "min_gain_to_split": kwargs.get("args").min_gain_to_split,
            "min_sum_hessian_in_leaf": kwargs.get("args").min_sum_hessian_in_leaf,
            "max_bin": kwargs.get("args").max_bin,
            "min_data_in_bin": kwargs.get("args").min_data_in_bin,
            "min_data_in_leaf": kwargs.get("args").min_data_in_leaf,
            "feature_fraction": kwargs.get("args").feature_fraction,
            "bagging_fraction": kwargs.get("args").bagging_fraction,
            "bagging_freq": kwargs.get("args").bagging_freq,
            "lambda_l1": kwargs.get("args").lambda_l1,
            "lambda_l2": kwargs.get("args").lambda_l2,
        }
        self.rum_structure = add_hyperparameters(self.rum_structure, hyperparameters)

        self.model_spec = {
            "rum_structure": self.rum_structure,
            "general_params": general_params,
        }

        # using gpu or not
        if kwargs.get("args").device == "cuda":
            self.torch_tensors = {"device": "cuda"}
        else:
            self.torch_tensors = None

    def build_dataloader(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame = None,
        y_valid: pd.Series = None,
        folds: list = None,
    ):
        """
        Builds and stores the LightGBM dataset.
        There is no specific dataloader for RUMBoost.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target variable.
        X_valid : pd.DataFrame, optional
            Validation features. The default is None.
        y_valid : pd.Series, optional
            Validation target variable. The default is None.
        folds : list, optional
            List of folds for cross-validation. The default is None.
            Can only specified if X_valid and y_valid are None.
        """
        if folds is not None and (X_valid is not None or y_valid is not None):
            raise ValueError(
                "If folds are specified, X_valid and y_valid cannot be specified."
            )
        if folds is not None:
            self.lgb_valid = []
            self.lgb_train = []
            for _, (train_idx, test_idx) in enumerate(folds):
                x_train_cv = X_train.iloc[train_idx]
                y_train_cv = y_train.iloc[train_idx]
                x_valid_cv = X_train.iloc[test_idx]
                y_valid_cv = y_train.iloc[test_idx]
                self.lgb_valid.append(
                    build_lgb_dataset(
                        x_valid_cv,
                        y_valid_cv,
                    )
                )
                self.lgb_train.append(
                    build_lgb_dataset(
                        x_train_cv,
                        y_train_cv,
                    )
                )
            self.lgb_train_full = build_lgb_dataset(
                X_train,
                y_train,
            )
        else:
            self.lgb_train = build_lgb_dataset(
                X_train,
                y_train,
            )
            if X_valid is not None and y_valid is not None:
                self.lgb_valid = build_lgb_dataset(
                    X_valid,
                    y_valid,
                )

    def fit(self):
        """
        Fits the model to the training data.
        """
        # cross-validation
        if isinstance(self.lgb_train, list):
            best_iteration = 0
            for i in range(len(self.lgb_train)):
                # train rumboost model
                model = rum_train(
                    self.lgb_train[i],
                    self.model_spec,
                    valid_sets=[self.lgb_valid[i]],
                    torch_tensors=self.torch_tensors,
                )
                best_iteration += model.best_iteration

            best_iteration /= len(self.lgb_train)
            self.model_spec["general_params"]["num_iterations"] = int(best_iteration)
            self.model_spec["general_params"]["early_stopping_rounds"] = None
            delattr(self, "lgb_valid")
            self.lgb_train = copy.deepcopy(self.lgb_train_full)

        # train rumboost model
        self.model = rum_train(
            self.lgb_train,
            self.model_spec,
            valid_sets=[self.lgb_valid] if hasattr(self, "lgb_valid") else None,
            torch_tensors=self.torch_tensors,
        )

        self.best_iteration = self.model.best_iteration

        return self.model.best_score_train

    def predict(self, X_test: pd.DataFrame) -> np.array:
        """
        Predicts the target variable for the test set.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test set.

        Returns
        -------
        preds : np.array
            Predicted target variable, as probabilities.
        binary_preds : np.array
            The binary probabilities of the target being bigger than each level.
        label_pred : np.array
            Predicted target variable, as labels.
        """
        assert hasattr(
            self, "model"
        ), "Model not trained yet. Please train the model before predicting."
        # build lgb dataset
        lgb_test = lgb.Dataset(X_test, free_raw_data=False)
        preds = self.model.predict(lgb_test)
        if self.torch_tensors:
            preds = preds.cpu().numpy()
        return preds

    def save_model(self, path: str):
        """
        Saves the model to the specified path.

        Parameters
        ----------
        path : str
            Path to save the model.
        """
        assert hasattr(
            self, "model"
        ), "Model not trained yet. Please train the model before saving."
        self.model.save_model(path)

    def load_model(self, path: str):
        """
        Loads the model from the specified path.

        Parameters
        ----------
        path : str
            Path to load the model from.
        """
        self.model = load_rumboost(model_file=path)


class APLR:
    """
    Wrapper class for APLR model.
    """

    def __init__(self, **kwargs):
        self.model = APLRClassifier(
            max_interaction_level=0,
            early_stopping_rounds=kwargs.get("args").early_stopping_rounds,
        )

        self.monotone_constraints = kwargs.get("monotone_constraints")

    def build_dataloader(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs,
    ):
        """
        Builds and stores the LightGBM dataset.
        There is no specific dataloader for APLR.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target variable.
        kwargs : dict
            Additional arguments. Not used in this case.
        """
        self.X_train = X_train
        self.y_train = y_train

    def fit(self):
        """
        Fits the model to the training data.
        """
        mc = [self.monotone_constraints[c] for c in self.X_train.columns]
        self.model.fit(self.X_train, self.y_train, monotonic_constraints=mc)

        preds = self.model.predict_class_probabilities(self.X_train)
        train_loss = cross_entropy(preds, self.y_train.values)
        return train_loss

    def predict(self, X_test: pd.DataFrame) -> np.array:
        """
        Predicts the target variable for the test set.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test set.

        Returns
        -------
        preds : np.array
            Predicted target variable, as probabilities.
        """
        assert hasattr(
            self, "model"
        ), "Model not trained yet. Please train the model before predicting."
        preds = self.model.predict_class_probabilities(X_test)
        return preds

    def save_model(self, path: str):
        """
        Saves the model to the specified path.

        Parameters
        ----------
        path : str
            Path to save the model.
        """
        raise NotImplementedError(
            "APLR model cannot be saved. Please use the APLRClassifier class directly."
        )

    def load_model(self, path: str):
        """
        Loads the model from the specified path.
        """
        raise NotImplementedError(
            "APLR model cannot be loaded. Please use the APLRClassifier class directly."
        )
