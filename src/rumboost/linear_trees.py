import json
import numpy as np


class LinearTree:
    def __init__(self, x, monotonic_constraint=0, max_bins=255, learning_rate=0.1):
        """x must be 1d shape"""
        self.bin_edges, self.histograms, self.bin_indices = (
            self.build_lightgbm_style_histogram(x, max_bins)
        )
        self.bin_indices = self.bin_indices.reshape(1, -1).repeat(
            self.bin_edges.shape[0] - 2, axis=0
        )
        self.x = x
        self.monotonic_constraint = monotonic_constraint
        self.upper_bound = 0
        self.lower_bound = 0
        self.x_minus_bin_edges = self.x[None, :] - self.bin_edges[1:-1, None]
        self.split_and_leaf_values = {
            "splits": np.array([x.min(), x.max()]),
            "constants": np.array([0.0, 0.0]),
            "leaves": np.array([0.0]),
            "value_at_splits": np.array([0.0, 0.0]),
        }
        self.feature_importance_dict = {"gain": np.array([])}
        self.valid_sets = []
        self.name_valid_sets = []
        self.learning_rate = learning_rate
        # x_0 - e_0, x_0 - e_1, ...
        # x_1 - e_0, x_1 - e_1, ...

    def build_lightgbm_style_histogram(self, feature_values, max_bins=255):

        percentiles = np.linspace(0, 100, max_bins + 1)
        bin_edges = np.unique(np.percentile(feature_values, percentiles))

        bin_indices = np.digitize(feature_values, bins=bin_edges[1:-1], right=True)

        histogram = np.bincount(bin_indices, minlength=len(bin_edges) - 1)

        return bin_edges, histogram, bin_indices

    def feature_importance(self, type: str):
        return self.feature_importance_dict[type]

    def update(self, train_set, fobj):

        grad, hess = fobj(1, 2)
        grad_x = grad * self.x_minus_bin_edges
        hess_x = hess * self.x_minus_bin_edges

        N = self.bin_indices.max() + 1
        id = self.bin_indices + (N * np.arange(self.bin_indices.shape[0]))[:, None]

        grad_x_binned = np.bincount(id.ravel(), weights=grad_x.ravel()).reshape(-1, N)
        hess_x_binned = np.bincount(id.ravel(), weights=hess_x.ravel()).reshape(-1, N)

        arange = np.arange(grad_x_binned.shape[1])
        edgerange = np.arange(grad_x_binned.shape[1] - 1) + 1
        mask = arange[None, :] < edgerange[:, None]

        left_gain = (grad_x_binned * mask).sum(axis=0) ** 2 / (
            hess_x_binned * mask
        ).sum(axis=0)
        left_leaf = -(grad_x_binned * mask).sum(axis=0) / (hess_x_binned * mask).sum(
            axis=0
        )

        left_gain = np.nan_to_num(left_gain, nan = -np.inf)

        right_gain = (grad_x_binned * ~mask).sum(axis=0) ** 2 / (
            hess_x_binned * ~mask
        ).sum(axis=0)
        right_leaf = -(grad_x_binned * ~mask).sum(axis=0) / (hess_x_binned * ~mask).sum(
            axis=0
        )
        right_gain = np.nan_to_num(right_gain, nan = -np.inf)

        no_split_gain = grad_x_binned.sum() / hess_x_binned.sum()

        if self.monotonic_constraint == 1:
            left_gain[left_leaf < self.lower_bound] = -np.inf
            right_gain[right_leaf < self.lower_bound] = -np.inf

        if self.monotonic_constraint == -1:
            left_gain[left_leaf > self.upper_bound] = -np.inf
            right_gain[right_leaf > self.upper_bound] = -np.inf

        gain = left_gain + right_gain - no_split_gain

        best_index = np.argmax(gain)
        self.best_split = self.bin_edges[best_index + 1]
        self.best_left_leaf = self.learning_rate * left_leaf[best_index]
        self.best_right_leaf = self.learning_rate * right_leaf[best_index]
        best_gain = gain[best_index]
        self.feature_importance_dict["gain"] = np.concatenate(
            [self.feature_importance_dict["gain"], np.array([best_gain])], axis=0
        )

    def rollback_one_iter(self):
        self.best_split = None
        self.best_left_leaf = None
        self.best_right_leaf = None
        self.feature_importance["gain"] = self.feature_importance["gain"][:-1]

    def _update_linear_constants(self):
        """ """
        if (
            self.best_split is None
            or self.best_left_leaf is None
            or self.best_right_leaf is None
        ):
            return 0

        s = self.best_split
        l_0 = self.best_left_leaf
        l_1 = self.best_right_leaf
        if (
            s in self.split_and_leaf_values["splits"]
        ):  # if the split value exists already
            index = np.searchsorted(self.split_and_leaf_values["splits"], s)
            self.split_and_leaf_values["leaves"][:index] += l_0

            self.split_and_leaf_values["leaves"][index:] += l_1

            self.split_and_leaf_values["constants"] = np.concatenate(
                (
                    self.split_and_leaf_values["constants"][: index + 1] + l_1 * self.split_and_leaf_values["splits"],
                    self.split_and_leaf_values["constants"][index + 1 :] + l_0 * self.split_and_leaf_values["splits"][index+1:],
                )
            )
        else:
            index = np.searchsorted(self.split_and_leaf_values["splits"], s)
            self.split_and_leaf_values["splits"] = np.insert(
                self.split_and_leaf_values["splits"], index, s
            )

            self.split_and_leaf_values["leaves"] = np.concatenate(
                (
                    self.split_and_leaf_values["leaves"][:index] + l_0,
                    self.split_and_leaf_values["leaves"][index - 1 :] + l_1,
                )
            )

            self.split_and_leaf_values["constants"] = np.concatenate(
                (
                    self.split_and_leaf_values["constants"][: index + 1] + l_1 * s,
                    self.split_and_leaf_values["constants"][index:] + l_0 * s,
                )
            )

        self.upper_bound = np.max(self.split_and_leaf_values["leaves"])
        self.lower_bound = np.min(self.split_and_leaf_values["leaves"])

        leaves = self.split_and_leaf_values["leaves"]
        splits = self.split_and_leaf_values["splits"]
        all_leaves = np.concatenate((leaves[0].reshape(-1), leaves))
        constants = self.split_and_leaf_values["constants"]

        self.split_and_leaf_values["value_at_splits"] = all_leaves * splits + constants

    def eval_train(self, feval):
        return [0]

    def eval_valid(self, feval):
        return [0]

    def model_to_string(self, _, __, ___) -> str:
        """
        Serialize the model to a JSON string.
        """
        model_dict = {
            "bin_edges": self.bin_edges.tolist(),
            "histograms": self.histograms.tolist(),
            "split_and_leaf_values": {
                k: v.tolist() for k, v in self.split_and_leaf_values.items()
            },
            "monotonic_constraint": self.monotonic_constraint,
            "upper_bound": self.upper_bound,
            "lower_bound": self.lower_bound,
            "feature_importance": self.feature_importance,
        }
        return json.dumps(model_dict)

    def model_from_string(self, s: str):
        """
        Load the model from a JSON string.
        """
        model_dict = json.loads(s)
        self.bin_edges = np.array(model_dict["bin_edges"])
        self.histograms = np.array(model_dict["histograms"])
        self.split_and_leaf_values = {
            k: np.array(v) for k, v in model_dict["split_and_leaf_values"].items()
        }
        self.monotonic_constraint = model_dict["monotonic_constraint"]
        self.upper_bound = model_dict["upper_bound"]
        self.lower_bound = model_dict["lower_bound"]
        self.feature_importance = model_dict["feature_importance"]

    def free_dataset(
        self,
    ):
        self.x = None

    def set_train_data_name(self, name: str) -> "LinearTree":
        """Set the name to the training Dataset.

        Parameters
        ----------
        name : str
            Name for the training Dataset.

        Returns
        -------
        self : Booster
            Booster with set training Dataset name.
        """
        self._train_data_name = name
        return self

    def add_valid(self, data, name: str) -> "LinearTree":
        """Add validation data.

        Parameters
        ----------
        data : Dataset
            Validation data.
        name : str
            Name of validation data.

        Returns
        -------
        self : Booster
            Booster with set validation data.
        """
        if not isinstance(self.valid_sets, list):
            self.valid_sets = []
        if not isinstance(self.name_valid_sets, list):
            self.name_valid_sets = []
        self.valid_sets.append(data)
        self.name_valid_sets.append(name)
        return self
