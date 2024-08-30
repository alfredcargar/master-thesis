"""
Functionalities used for the EDA analysis and feature selection

Author: smas
Last update: 02/01/2024
"""

from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, r_regression
from sklearn.decomposition import PCA

from .configuration import *

# TODO: Add to the toolbox


def features_selection(df_features: pd.DataFrame, df_target: pd.DataFrame, method: str,
                       k: int = 10, color_bars_according_to_topic: bool =False,
                       figsize: Tuple[int, int] = (10, 10), rotation_labels: int = 90, verbose: bool = True):
    """
    This function performs a feature selection based on Univariate feature selection

    Args:
        df_features: DataFrame with all the features to be analyzed
        df_target: DataFrame with the target
        method: To chose between {f_regression, r_regression}
        k: Number of features to select for the analysis
        color_bars_according_to_topic: Indicates if the bars will be colored according to topic
        figsize: Size of the final image
        rotation_labels: Rotation of the x-labels

    Returns:
        DataFrame with the selected features after the analysis

    Notes:
        1. https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
        2. Univariate feature selection works by selecting the best features based on univariate statistical tests.
        3. f_regression -> [f_statistic, p_values]
            a. f_statistic: The F-statistic is a measure of how well the linear regression model you're fitting
                explains the variance in the target variable (dependent variable). It is a ratio of two variances:
                the variance explained by your model and the variance not explained by your model.A higher F-statistic
                indicates that your model does a better job of explaining the variance in the target variable
            b. The p-values associated with each feature are a measure of the statistical significance of that
                feature in the regression model. A small p-value (typically less than a chosen significance level,
                such as 0.05) suggests that the feature is statistically significant in predicting the target variable.
                In other words, it indicates that there is evidence to reject the null hypothesis
        4. r_regression -> correlation_coefficient
            a. correlation_coefficient: Pearson’s R correlation coefficients of features. It measures the strength
                and direction of the linear association between two variables. The Pearson correlation coefficient can
                take values between -1 and 1, with the following interpretations:
                    i. r< 0:  As one variable increases, the other tends to decrease
                    ii. r = 0: There is no linear relationship between the variables
                    iii. r > 0: As one variable increases, the other tends to increase
    """

    if method == 'f_regression':
        selector = SelectKBest(f_regression, k=k)
    elif method == 'r_regression':
        selector = SelectKBest(r_regression, k=k)
    elif method == 'f_classif':
        selector = SelectKBest(f_classif, k=k)
    else:
        raise ValueError(f"Method: {method} invalid. Select: {'r_regression', 'f_regression'}")

    # Perform the feature selection
    selector.fit_transform(df_features, df_target)
    # Obtain the score of each feature
    scores = selector.scores_
    # Obtain the indexes of the k best features, regardless of the sign
    idxs_max_score = np.abs(scores).argsort()[-k:][::-1]
    # Obtain the k best scores
    selected_scores = scores[idxs_max_score]

    # Obtain the name of the k best features
    name_features = np.array(list(df_features.columns))
    selected_features = name_features[idxs_max_score]

    # Improve visualization
    if method == 'f_regression' or method == 'f_classif':
        selected_scores = np.log10(selected_scores)
    selected_scores = selected_scores

    # Plot the results
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    if method == 'f_regression':
        plt.ylabel('log(F-Statistics scores)', fontsize=16)
    else:
        plt.ylabel('Pearson’s R correlation coefficients', fontsize=16)

    # Color the bars according to the topic
    if color_bars_according_to_topic:
        colors, features_type_available = _define_colors_according_to_features_topic(selected_features)
        ax.bar(selected_features[0:k], selected_scores[0:k], color=colors)
        legend_elements = _define_patches_features_type_available(features_type_available)
        if len(legend_elements) == 5:
            ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0., 1.02, 1., .102),
                      fontsize=14, ncol=len(legend_elements), mode="expand", borderaxespad=0.)
        else:
            ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0., 1.02, 1., .102),
                      fontsize=14, ncol=len(legend_elements), mode=None, borderaxespad=0.)
    else:
        ax.bar(selected_features[0:k], selected_scores[0:k])

    fig.subplots_adjust(bottom=0.5)
    plt.xticks(fontsize=12, rotation=rotation_labels)
    return selected_features, selected_scores, fig


def _define_colors_according_to_features_topic(selected_features):
    colors = []
    features_type_available = []

    for feature in selected_features:
        if feature in SECTOR_GEOMETRY:
            features_type_available.append("geometry")
            colors.append('skyblue')
        elif feature in FLUXES:
            features_type_available.append("fluxes")
            colors.append('mediumpurple')
        elif feature in SPATIAL_LOCATION:
            features_type_available.append("spatial")
            colors.append('darkgreen')
        elif feature in FLUXES_INTERACTION:
            features_type_available.append("interaction")
            colors.append('orange')
        elif feature in COMPLEXITY:
            features_type_available.append("complexity")
            colors.append('blue')
        elif feature in OC_EC:
            features_type_available.append("oc_ec")
            colors.append('firebrick')
        else:
            raise ValueError(f" no predefined topic for the features {feature}")

    return colors, features_type_available


def _define_patches_features_type_available(features_type_available):
    legend_elements = []

    if "geometry" in features_type_available:
        # legend_elements.append(Patch(facecolor='skyblue', edgecolor='skyblue', label='Operational \ntime'))
        legend_elements.append(Patch(facecolor='skyblue', edgecolor='skyblue', label='Geometry'))
    if "fluxes" in features_type_available:
        legend_elements.append(Patch(facecolor='mediumpurple', edgecolor='mediumpurple', label='Fluxes'))
    if "spatial" in features_type_available:
        legend_elements.append(Patch(facecolor='darkgreen', edgecolor='darkgreen', label='Spatial'))
    if "interaction" in features_type_available:
        legend_elements.append(Patch(facecolor='orange', edgecolor='orange', label='Interactions'))
    if "complexity" in features_type_available:
        legend_elements.append(Patch(facecolor='blue', edgecolor='blue', label='Complexity'))
    if "oc_ec" in features_type_available:
        legend_elements.append(Patch(facecolor='firebrick', edgecolor='firebrick', label='OC_EC'))

    return legend_elements


def pca(df_features: pd.DataFrame, variance_threshold: int = 0.95) -> Tuple[np.ndarray, PCA]:
    """
    Perform PCA on the features and return the new features and the PCA object

    Args:
        df_features (pd.DataFrame): features
        variance_threshold (int, optional): variance threshold. Defaults to 0.95.

    Returns:
        Tuple[np.ndarray, PCA]: new features and PCA object

    Notes:
        1. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA

    """

    selector = PCA()
    new_features = selector.fit_transform(df_features)
    explained_variance = selector.explained_variance_ratio_
    cum_variance = 0
    counter = 0
    for i in explained_variance:
        cum_variance += i
        counter += 1
        if cum_variance >= variance_threshold:
            print("From ", new_features.shape[1], "to", counter, "features after PCA with ",
                  "{:.2f}".format(cum_variance), "variance retained.")
            break

    cum_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(6, 6))
    plt.plot(np.linspace(1, len(cum_variance), len(cum_variance)), cum_variance, '-o')
    plt.xlabel('Number of features', fontsize=11)
    plt.ylabel('Explained Variance', fontsize=11)
    plt.subplots_adjust(left=0.2)
    new_features = new_features[:, 0:counter + 1]

    return new_features, selector
