import statsmodels.api as sm
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def outliers_to_nan(var_num):
    """
    Detects outliers and replaces them with NaN, inplace
    - var: df variable (Series)
    Returns: 
    - Modified Series with outliers replaced by nan
    - Count of outliers
    """
    if (var_num.skew()) < 1:
        # symmetrical, use std
        c1 = abs((var_num - var_num.mean()) / var_num.std()) > 3
    else:
        # asymmetrical, use MAD
        mad = sm.robust.mad(var_num, axis=0)
        c1 = abs((var_num - var_num.median()) / mad) > 8

    qnt = var_num.quantile([0.25, 0.75]).dropna()
    Q1 = qnt.iloc[0]
    Q3 = qnt.iloc[1]
    delta = 3 * (Q3 -Q1)

    c2 = (var_num < (Q1 - delta)) | (var_num > (Q3 + delta))
    var = var_num.copy()
    var[c1 & c2] = np.nan
    return [var, sum(c1 & c2)]


def distinct_values(df):
    """
    Counts the unique values of the df numerical variables
    Args:
        An input dataframe
    Returns:
        a dataframe showing a count of distinct values for each variable
    """
    nums = df.select_dtypes(include=['int','int32','int64','float','float32','float64'])
    h = nums.apply(lambda x: len(x.unique()))
    output = pd.DataFrame({'variable': h.index, 'distinct values': h.values})
    return output

def fillna_on_skewness(df):
    """
    Applies imputation on nan values using the mean or median, depending on the skewness
    Args:
        df: dataset with only numerical columns
    Returns:
        df after nan imputation
    """
    for var in df.columns:
        skewness = df[var].skew()
        mean = df[var].mean()
        median = df[var].median()
        if (abs(skewness)) < 0.5:
            df[var].fillna(mean, inplace=True)
        else:
            df[var].fillna(median, inplace=True)
    return df

def v_cramer(v, target):
    """
    Calcula el coeficiente V de Cramer entre dos variables. Si alguna de ellas es continua, la discretiza.

    Datos de entrada:
    - v: Serie de datos categóricos o cuantitativos.
    - target: Serie de datos categóricos o cuantitativos.

    Datos de salida:
    - Coeficiente V de Cramer que mide la asociación entre las dos variables.
    """

    if v.dtype == 'float64' or v.dtype == 'int64':
        # Si v es numérica, la discretiza en intervalos y rellena los valores faltantes
        p = sorted(list(set(v.quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]))))
        v = pd.cut(v, bins=p)
        v = v.fillna(v.min())

    if target.dtype == 'float64' or target.dtype == 'int64':
        # Si target es numérica, la discretiza en intervalos y rellena los valores faltantes
        p = sorted(list(set(target.quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]))))
        target = pd.cut(target, bins=p)
        target = target.fillna(target.min())

    # Calcula una tabla de contingencia entre v y target
    tabla_cruzada = pd.crosstab(v, target)

    # Calcula el chi-cuadrado y el coeficiente V de Cramer
    chi2 = chi2_contingency(tabla_cruzada)[0]
    n = tabla_cruzada.sum().sum()
    v_cramer = np.sqrt(chi2 / (n * (min(tabla_cruzada.shape) - 1)))

    return v_cramer


def vcramer_plot(matriz, target):
    """
    Genera un gráfico de barras horizontales que muestra el coeficiente V de Cramer entre cada columna de matriz y la variable target.

    Datos de entrada:
    - matriz: DataFrame con las variables a comparar.
    - target: Serie de la variable objetivo (categórica).

    Datos de salida:
    - Gráfico de barras horizontales que muestra el coeficiente V de Cramer para cada variable.
    """

    # Calcula el coeficiente V de Cramer para cada columna de matriz y target
    salidaVcramer = {x: v_cramer(matriz[x], target) for x in matriz.columns}

    # Ordena los resultados en orden descendente por el coeficiente V de Cramer
    sorted_data = dict(sorted(salidaVcramer.items(), key=lambda item: item[1], reverse=True))

    # Crea el gráfico de barras horizontales
    plt.figure(figsize=(10, 8))
    plt.barh(list(sorted_data.keys()), list(sorted_data.values()), color='skyblue')
    plt.xlabel('V de Cramer')
    plt.tight_layout()
    plt.show()



def optimized_transf_cont(var, target):
    """
    Esta función busca la mejor transformación para una variable cuantitativa que maximice la correlación con una variable objetivo.

    Datos de entrada:
    - var: Serie de datos cuantitativos que se desea transformar.
    - target: Serie de la variable objetivo con la que se busca maximizar la correlación.

    Datos de salida:
    - Una lista que contiene el nombre de la mejor transformación y la serie transformada correspondiente.
    """

    # Normaliza la serie utilizando StandardScaler
    var = StandardScaler().fit_transform([[x] for x in list(var)])

    # Asegura que los valores sean positivos
    var = var + abs(np.min(var)) * 1.0001
    var = [x[0] for x in var]

    # Crea un DataFrame con posibles transformaciones de la variable
    candidates = pd.DataFrame({
        'x_': var,
        'logx_': np.log(var),
        'expx_': np.exp(var),
        'pow2x_': [x**2 for x in var],
        'sqrtx_': np.sqrt(var),
        'pow4x_': [x**4 for x in var],
        '4rtx_': [x**(1/4) for x in var]
    })

    # Calcula la correlación entre las transformaciones y la variable objetivo
    cor_values = candidates.apply(
        lambda col: np.abs(np.corrcoef(target, col, rowvar=False, ddof=0)[0, 1]),
        axis=0
    )

    # Encuentra la transformación con la correlación máxima
    max_corr_idx = cor_values.idxmax()

    return [max_corr_idx, candidates[max_corr_idx]]
    
    
   
def apply_transf(matriz, target):
    """
    Esta función realiza transformaciones automáticas en las columnas de una matriz de datos para maximizar la correlación (si target es numérica)
    o el coeficiente V de Cramer (si target es categórica) con una variable objetivo.
    

    Datos de entrada:
    - matriz: DataFrame con las variables a transformar.
    - target: Serie de la variable objetivo (numérica o categórica).

    Datos de salida:
    - DataFrame con las mejores transformaciones aplicadas a las columnas.
    """
     
    aux = matriz.apply(
            lambda col: optimized_transf_cont(col, target),
            axis=0
    )

    # Extrae las transformaciones óptimas y las series transformadas correspondientes
    aux2 = aux.apply(lambda col: col[1], axis=0)
    aux = aux.apply(lambda col: col[0], axis=0)

    # Renombra las columnas de aux2 con el nombre de las transformaciones
    aux2.columns = [aux[x] + aux2.columns[x] for x in range(len(aux2.columns))]

    # Asigna los índices de la matriz original a aux2
    aux2.index = matriz.index

    return aux2


def plot_features_imp(importances, x_train):
    # Sort the feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.title('Feature Importances')
    plt.bar(range(x_train.shape[1]), importances[indices], align='center', color='skyblue')
    plt.xticks(range(x_train.shape[1]), x_train.columns[indices], rotation=90)
    plt.xlim([-1, x_train.shape[1]])
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.show()







    