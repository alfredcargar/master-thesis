"""
Utilities related to pre-processing the data before starting the training

Author: smas
Last update: 03/01/2024
"""

import numpy as np
import pandas as pd


def capacity_for_classification(target_values: np.array, lower_bound: float, upper_bound:float, step_size: int) -> np.array:
    """
    This function transforms the capacity target values to into a classification problem

    Args:
        target_values: Array containing the target values
        lower_bound: Lower bound of the capacity
        upper_bound: Upper bound of the capacity
        step_size: Step size

    Returns:
        Array containing the target values transformed into a classification problem based on the step size
    """

    class_boundaries = list(range(int(lower_bound), int(upper_bound), step_size))

    class_values = []

    for value in target_values:
        if value < class_boundaries[0]:
            class_values.append(0)
        elif value >= class_boundaries[-1]:
            class_values.append(len(class_boundaries))
        else:
            for i in range(len(class_boundaries) - 1):
                if class_boundaries[i] <= value < class_boundaries[i + 1]:
                    class_values.append(i + 1)

    return np.array(class_values)


def train_test_split_by_sector_name(df_dataset, test_size):
    """
    This function splits the dataset into train and test sets by sector name

    Args:
        df_dataset: Dataset to split
        test_size: Size of the test set

    Returns:
        df_train: Training dataset
        df_test: Test dataset
    """

    assert 1 > test_size > 0, f"Test size must be between 0 and 1. It is {test_size}"

    df_train_size = np.ceil(len(df_dataset) * (1-test_size))

    unique_sector_code = np.unique(df_dataset['SectorCode'].values)

    df_train = pd.DataFrame()
    while len(df_train) < df_train_size:
        random_sector_code_idx = np.random.randint(0, len(unique_sector_code), 1)[0]
        sector_code_to_remove_from_train = unique_sector_code[random_sector_code_idx]

        df_train_random_sector_code = df_dataset[df_dataset['SectorCode'] == sector_code_to_remove_from_train]
        df_train = pd.concat([df_train, df_train_random_sector_code])

    list_sector_codes_df_train = list(df_train['SectorCode'].values)

    df_test = df_dataset[~df_dataset['SectorCode'].isin(list_sector_codes_df_train)]

    return df_train, df_test


def train_test_split_temporal(df_dataset, column_sort, test_size):
    # TODO: Typing and documentation

    assert column_sort in df_dataset.columns.values

    df_dataset_sorted = df_dataset.sort_values(column_sort)

    num_observations_train = len(df_dataset_sorted) * (1 - test_size)
    if num_observations_train.is_integer():
        num_observations_train = int(num_observations_train)
        num_observations_test = int(len(df_dataset_sorted) - num_observations_train)
    else:
        num_observations_train = int(round(num_observations_train))
        num_observations_test = int(len(df_dataset_sorted) - num_observations_train)

    df_train = df_dataset_sorted.iloc[0:num_observations_train]
    df_test = df_dataset_sorted.iloc[num_observations_train:]

    return df_train, df_test


def features_target_split(df: pd.DataFrame, target_name: str = 'target') -> (pd.DataFrame, pd.DataFrame):
    """
    This function extracts the target from a dataset

    Args:
        df: DataFrame containing features and target
        target_name: Name of the target column

    Returns:
        df containing features and df containing target
    """

    # TODO: df_target is not a dataframe. It is an array. Make sure it is a dataframe

    df_target = np.reshape(np.array(pd.DataFrame(df[target_name], index=df.index, columns=[target_name])), -1)
    # df_target = df[target_name]
    df_feat = df.drop(columns=[target_name])

    return df_feat, df_target