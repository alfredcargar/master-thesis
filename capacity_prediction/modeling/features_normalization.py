"""
Utilities used to normalize the features

Author: smas
Last update: 06/10/2023
"""

import pandas as pd
from sklearn import preprocessing


def normalize_dataset(df_features: pd.DataFrame, norm: str = 'l1', axis: int = 1) -> pd.DataFrame:
    """
    Normalization is the process of scaling individual samples to have unit norm

    Args:
        df_features: DataFrame containing features
        norm: The norm to use to normalize
        axis: If 1, independently normalize each sample, otherwise (if 0) normalize each feature.

    Returns:
        df_features normalized

    Notes:
        1. This process can be useful if you plan to use a quadratic form such as the dot-product or
            any other kernel to quantify the similarity of any pair of samples. E.g., Gradient-Based Models (NN),
            Distance-Based Models (KNN, SVM, or K-Means), Regularized Linear Models (Linear Regression, Logistic Regression),
            Clustering Algorithms (K-Means), or Anomaly Detection (One-Class SVM), PCA, RNNs
        2. Tree-Based Models (Random Forests or Decision trees) don't require normalization
        3. Gradient Boosting Machines (e.g., XGBoost, LightGBM): While gradient boosting algorithms are not as sensitive
            to feature scaling as some other models, normalizing input features can still improve performance.
    """

    assert norm in ['l1', 'l2', 'max'], f"Norm {norm} not supported. Use: l1, l2, or max"
    assert axis in [0, 1], f"Axis {axis} not supported. Use: 0 (each feature) or 1 (each sample)"

    normalized_array = preprocessing.normalize(df_features, norm=norm, axis=axis)
    df_features_normalized = pd.DataFrame(normalized_array, columns=df_features.columns)

    return df_features_normalized
