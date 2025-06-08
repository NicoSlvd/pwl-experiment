import biogeme.database as db
import pandas as pd
from biogeme.expressions import Beta
from biogeme.models import loglogit
import biogeme.biogeme as bio


def SwissMetro_normalised(dataset_train: pd.DataFrame, for_prob=False):
    """
    Create a MNL on the swissmetro dataset.

    Parameters
    ----------
    dataset_train : pandas DataFrame
        The training dataset.

    Returns
    -------
    biogeme : bio.BIOGEME
        The BIOGEME object containing the model.
    """
    database_train = db.Database("swissmetro_train", dataset_train)

    globals().update(database_train.variables)

    # parameters to be estimated
    ASC_CAR = Beta("ASC_CAR", 0, None, None, 0)
    ASC_SM = Beta("ASC_SM", 0, None, None, 0)
    ASC_TRAIN = Beta("ASC_TRAIN", 0, None, None, 1)

    B_TIME_CAR = Beta("B_TIME_CAR", 0, None, 0, 0)
    B_COST_CAR = Beta("B_COST_CAR", 0, None, 0, 0)
    B_TIME_RAIL = Beta("B_TIME_RAIL", 0, None, 0, 0)
    B_COST_RAIL = Beta("B_COST_RAIL", 0, None, 0, 0)
    B_HE_RAIL = Beta("B_HE_RAIL", 0, None, 0, 0)
    B_TIME_SM = Beta("B_TIME_SM", 0, None, 0, 0)
    B_COST_SM = Beta("B_COST_SM", 0, None, 0, 0)
    B_HE_SM = Beta("B_HE_SM", 0, None, 0, 0)

    B_FIRST_SM = Beta("B_FIRST_SM", 0, None, None, 0)
    B_MALE_SM = Beta("B_MALE_SM", 0, None, None, 0)
    B_FIRST_CAR = Beta("B_FIRST_CAR", 0, None, None, 0)
    B_MALE_CAR = Beta("B_MALE_CAR", 0, None, None, 0)

    B_PURPOSE_1_SM = Beta("B_PURPOSE_1_SM", 0, None, None, 0)
    B_PURPOSE_2_SM = Beta("B_PURPOSE_2_SM", 0, None, None, 0)
    B_PURPOSE_3_SM = Beta("B_PURPOSE_3_SM", 0, None, None, 0)
    B_PURPOSE_4_SM = Beta("B_PURPOSE_4_SM", 0, None, None, 0)
    B_PURPOSE_1_CAR = Beta("B_PURPOSE_1_CAR", 0, None, None, 0)
    B_PURPOSE_2_CAR = Beta("B_PURPOSE_2_CAR", 0, None, None, 0)
    B_PURPOSE_3_CAR = Beta("B_PURPOSE_3_CAR", 0, None, None, 0)
    B_PURPOSE_4_CAR = Beta("B_PURPOSE_4_CAR", 0, None, None, 0)

    B_AGE_1_SM = Beta("B_AGE_1_SM", 0, None, None, 0)
    B_AGE_2_SM = Beta("B_AGE_2_SM", 0, None, None, 0)
    B_AGE_1_CAR = Beta("B_AGE_1_CAR", 0, None, None, 0)
    B_AGE_2_CAR = Beta("B_AGE_2_CAR", 0, None, None, 0)

    # utilities
    V_TRAIN = (
        ASC_TRAIN
        + B_TIME_RAIL * TRAIN_TT_constant
        + B_COST_RAIL * TRAIN_COST_constant
        + B_HE_RAIL * TRAIN_HE_constant
    ) 
    V_SM = (
        ASC_SM
        + B_TIME_SM * SM_TT_constant
        + B_COST_SM * SM_COST_constant
        + B_HE_SM * SM_HE_constant
        + B_FIRST_SM * FIRST_constant
        + B_MALE_SM * MALE_constant
        + B_PURPOSE_1_SM * PURPOSE_1_constant
        + B_PURPOSE_2_SM * PURPOSE_2_constant
        + B_PURPOSE_3_SM * PURPOSE_3_constant
        + B_PURPOSE_4_SM * PURPOSE_4_constant
        + B_AGE_1_SM * AGE_1_constant
        + B_AGE_2_SM * AGE_2_constant
    )  
    V_CAR = (
        ASC_CAR 
        + B_TIME_CAR * CAR_TT_constant 
        + B_COST_CAR * CAR_CO_constant
        + B_FIRST_CAR * FIRST_constant
        + B_MALE_CAR * MALE_constant
        + B_PURPOSE_1_CAR * PURPOSE_1_constant
        + B_PURPOSE_2_CAR * PURPOSE_2_constant
        + B_PURPOSE_3_CAR * PURPOSE_3_constant
        + B_PURPOSE_4_CAR * PURPOSE_4_constant
        + B_AGE_1_CAR * AGE_1_constant
        + B_AGE_2_CAR * AGE_2_constant
    )  

    V = {0: V_TRAIN, 1: V_SM, 2: V_CAR}
    av = {0: 1, 1: 1, 2: 1}

    # choice model
    logprob = loglogit(V, av, choice)
    biogeme = bio.BIOGEME(database_train, logprob)
    biogeme.modelName = "SwissmetroMNL~"

    biogeme.generate_html = True
    biogeme.generate_pickle = True

    if for_prob:
        util_spec = {
            "0": V[0],
            "1": V[1],
            "2": V[2],
        }
        biosim = bio.BIOGEME(database_train, util_spec)
        biosim.modelName = "swissmetro_logit_test"

        biosim.generate_html = False
        biosim.generate_pickle = False

        return biosim

    return biogeme