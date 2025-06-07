import random
import pickle
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Union

import lightgbm as lgb
import pandas as pd
import numpy as np

# type hints for dataset split function
TrainTestSplit = Union[
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],  # train and test
    tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],  # train, val and test
]


def augment_dataset(
    data: pd.DataFrame, feature_names: list[str], type: str = "cubic"
) -> pd.DataFrame:
    """
    Augment the dataset with specified polynomial features.

    Parameters
    ----------
    data : pd.DataFrame
        The input data.
    feature_names : list[str]
        The names of the features to be used as cubic polynomials in the model.
    type : str
        The type of polynomial features to be generated.
        Can be "cubic", "linear", "constant", "constant_linear", "constant_cubic".
    """
    new_data = pd.DataFrame({})
    # Generate cubic polynomial features
    for feature in feature_names:
        if "constant" in type:
            new_data[f"{feature}_constant"] = data[feature]
        if "linear" in type or "cubic" in type:
            new_data[f"{feature}_linear"] = data[feature]
        if "cubic" in type:
            new_data[f"{feature}_square"] = data[feature] ** 2
            new_data[f"{feature}_cubic"] = data[feature] ** 3

    return new_data


def generate_rum_structure(
    structure: Dict[int, List[str]],
    monotone_constraints: Dict[int, List[int]] = None,
    init_leaf_val: Optional[Dict[int, Dict[str, float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate the rum structure for the given dataset.

    Parameters
    ----------
    structure: Dict[int, List[str]]
        The structure of the rum model.
        The keys are the ensemble indices and the values are the feature names.
    monotone_constraints: Dict[int, List[int]], optional
        The monotone constraints for the rum model.
        The keys are the ensemble indices and the values are the monotone constraints.
        The default is None.
    init_leaf_val: Optional[Dict[int, Dict[str, float]]], optional
        The initial leaf values for the rum model.

    Returns
    -------
    rum_structure: List[Dict[Any]]
        The rum structure for the RUMBoost model.

    """

    # initialise rum_structure
    rum_structure = []

    # alternative-specific features, one per ensemble
    for u_idx in structure.keys():
        for i, f in enumerate(structure[u_idx]):
            # add the feature to the rum structure
            rum_structure.append(
                {
                    "variables": [f],
                    "utility": [u_idx],
                    "boosting_params": {
                        "monotone_constraints_method": "advanced",
                        "max_depth": 1,
                        "n_jobs": -1,
                        "learning_rate": 0.1,
                        "verbose": -1,
                        "monotone_constraints": (
                            [monotone_constraints[f]]
                            if monotone_constraints
                            else [0]
                        ),
                    },
                    "shared": False,
                }
            )

    return rum_structure

def generate_boost_from_param_space(
        rum_structure: List[Dict[str, Any]],
        cont_vars: List[str],
) -> List[str]:
    """
    Generate the list stating if a parameter is boosted from parameter space.

    Parameters
    ----------
    rum_structure: List[Dict[str, Any]]
        The rum structure for the RUMBoost model.
    cont_vars: List[str]
        The continuous variables in the dataset.

    Returns
    -------
    boosting_params: List[str]
        The boosting parameters list.

    """
    boosting_params = []
    for struct in rum_structure:
        # check if the feature is continuous or binary
        if struct["variables"][0] in cont_vars and "constant" not in struct["variables"][0]:
            boosting_params.append(True)
        else:
            boosting_params.append(False)

    return boosting_params

def add_hyperparameters(
    rum_struct: List[Dict[str, Any]],
    hyperparameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Add hyperparameters to a specific dict of rum structure.

    Parameters
    ----------
    rum_struct: List[Dict[str, Any]]
        The rum structure to be modified.
    hyperparameters: Dict[str, Any]
        The hyperparameters to be added to the rum structure.

    Returns
    -------
    rum_structure: List[Dict[Any]]
        The modified rum structure with the hyperparameters added.

    """
    for struct in rum_struct:
        # add the hyperparameters to the rum structure
        struct["boosting_params"].update(hyperparameters)

    return rum_struct


def generate_general_params(num_classes: int, **kwargs) -> Dict[str, Any]:
    """ "
    Generate the general parameters for the rumboost model.

    Parameters
    ----------
    num_classes: int
        The number of classes in the dataset.
    kwargs: Dict[str, Any]
        The additional parameters to be added to the general parameters.
        These parameters will be used to update the general parameters.
        It has to be parameters that are accepted by rumboost.
        See the rumboost documentation for more details.

    Returns
    -------
    general_params: Dict[str, Any]
        The general parameters for the rumboost model.
    """
    # general parameters
    general_params = {
        "num_classes": num_classes,
        "max_booster_to_update": 1,
    }

    # update the general parameters with the kwargs
    general_params.update(kwargs)

    return general_params


def build_lgb_dataset(X: pd.DataFrame, y: pd.Series) -> lgb.Dataset:
    """
    Build the LightGBM dataset from the dataframe.

    Parameters
    ----------
    X: pd.DataFrame
        The dataframe to be used.
    y: pd.Series
        The target variable.

    Returns
    -------
    lgb_dataset: Any
        The LightGBM dataset.
    """

    # create the LightGBM dataset
    lgb_dataset = lgb.Dataset(X, label=y, free_raw_data=False)

    return lgb_dataset

def transform_vars_list(
    vars_list: List[str], model_type: str, cont_vars: List[str]
) -> List[str]:
    """
    Transform the variable names in structure to a specific format.

    Parameters
    ----------
    vars_list: List[str]
        The list of variable names to be transformed.
    model_type: str
        The type of the model to be used.
        Can be "constant", "linear", "cubic", "constant_linear", "constant_cubic".
    cont_vars: List[str]
        The list of continuous variables in the dataset.
        This is used to check if the variable is continuous or not.
        If the variable is continuous, it will be transformed to a specific format.

    Returns
    -------
    new_vars_list: List[str]
        The transformed list of variable names.
    """
    if model_type == "constant":
        new_vars_list = [f"{var}_constant" for var in vars_list if var in cont_vars]
    elif model_type == "constant_linear":
        new_vars_list = [f"{var}_constant" for var in vars_list if var in cont_vars]
        new_vars_list += [f"{var}_linear" for var in vars_list if var in cont_vars]
    elif model_type == "constant_cubic":
        new_vars_list = [f"{var}_constant" for var in vars_list if var in cont_vars]
        new_vars_list += [f"{var}_linear" for var in vars_list if var in cont_vars]
        new_vars_list += [f"{var}_square" for var in vars_list if var in cont_vars]
        new_vars_list += [f"{var}_cubic" for var in vars_list if var in cont_vars]
    elif model_type == "linear":
        new_vars_list = [f"{var}_linear" for var in vars_list if var in cont_vars]
    elif model_type == "cubic":
        new_vars_list = [f"{var}_linear" for var in vars_list if var in cont_vars]
        new_vars_list += [f"{var}_square" for var in vars_list if var in cont_vars]
        new_vars_list += [f"{var}_cubic" for var in vars_list if var in cont_vars]

    return new_vars_list

def transform_mono_cons(
        mono_cons: Dict[str, int], model_type: str, cont_vars: List[str]
) -> Dict[str, int]:
    """
    Transform the monotone constraints to a specific format.

    Parameters
    ----------
    mono_cons: Dict[str, int]
        The monotone constraints to be transformed.
    model_type: str
        The type of the model to be used.
        Can be "constant", "linear", "cubic", "constant_linear", "constant_cubic".
    cont_vars: List[str]
        The list of continuous variables in the dataset.
        This is used to check if the variable is continuous or not.
        If the variable is continuous, it will be transformed to a specific format.

    Returns
    -------
    new_mono_cons: Dict[str, int]
        The transformed monotone constraints.
    """
    new_mono_cons = {}
    for var in mono_cons.keys():
        if var in cont_vars:
            if model_type == "constant":
                new_mono_cons[f"{var}_constant"] = mono_cons[var]
            elif model_type == "constant_linear":
                new_mono_cons[f"{var}_constant"] = mono_cons[var]
                new_mono_cons[f"{var}_linear"] = mono_cons[var]
            elif model_type == "constant_cubic":
                new_mono_cons[f"{var}_constant"] = mono_cons[var]
                new_mono_cons[f"{var}_linear"] = mono_cons[var]
                new_mono_cons[f"{var}_square"] = mono_cons[var]
                new_mono_cons[f"{var}_cubic"] = mono_cons[var]
            elif model_type == "linear":
                new_mono_cons[f"{var}_linear"] = mono_cons[var]
            elif model_type == "cubic":
                new_mono_cons[f"{var}_linear"] = mono_cons[var]
                new_mono_cons[f"{var}_square"] = mono_cons[var]
                new_mono_cons[f"{var}_cubic"] = mono_cons[var]
        else:
            new_mono_cons[f"{var}_constant"] = mono_cons[var]

    return new_mono_cons