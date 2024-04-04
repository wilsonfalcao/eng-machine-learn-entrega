from typing import Dict, Tuple

import pandas as pd
from pyspark.sql import Column
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import DoubleType


# Importação 02
from sklearn.model_selection import train_test_split
from sklearn import tree, preprocessing, metrics, model_selection, linear_model
from sklearn import model_selection
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

mlflow.end_run()
experiment_name = 'eng_machine_learn_entrega'

experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)
experiment_id = experiment.experiment_id

# Para o pré-processamento de dados
import numpy as np

# Para data analise e visualização
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


def _is_true(x: Column) -> Column:
    return x == "t"


def _parse_percentage(x: Column) -> Column:
    x = regexp_replace(x, "%", "")
    x = x.cast("float") / 100
    return x


def _parse_money(x: Column) -> Column:
    x = regexp_replace(x, "[$£€]", "")
    x = regexp_replace(x, ",", "")
    x = x.cast(DoubleType())
    return x
    
def load_kobe_dataset(kobe: pd.DataFrame) -> pd.DataFrame:
    """Load shuttles to csv because it's not possible to load excel directly into spark.
    """
    print("Loading data kobe data set...")
    kobe_data_raw = kobe
    kobe_data_raw.dtypes

    #drop rows with na values in the target feature and reset the index so we dont have anything missing
    kobe_data_raw = kobe_data_raw[kobe_data_raw['shot_made_flag'].notnull()].reset_index() 

    # Alterando os tipos de variáveis para melhor performance
    kobe_data_raw["period"] = kobe_data_raw["period"].astype('object')
    
    vars_to_category = ["combined_shot_type", "game_event_id", "game_id", "playoffs",
                        "season", "shot_made_flag", "shot_type", "team_id"]
    for col in vars_to_category:
        kobe_data_raw[col] = kobe_data_raw[col].astype('category')

    kobe_data_raw.describe(include=['number'])
    kobe_data_raw['shot_made_flag'] = kobe_data_raw['shot_made_flag'].astype(int)
    
    return kobe_data_raw
    
def kobe_avarage_shot_artefact(kobe_shot: pd.DataFrame) -> pd.DataFrame:

        figure_name = 'kobe_avarage_shot_artefact'
        data = kobe_shot
        sns.set(style='ticks')
            
        #percentage = [ (made shots) / (total shots) ] * 100
        made_shots_num = data['shot_made_flag'].value_counts()[1]
        total_shots = data['shot_made_flag'].value_counts()[0]+ data['shot_made_flag'].value_counts()[1]
        fg_percentage = 100 * made_shots_num / total_shots
        sns.countplot(data=data,x='shot_made_flag',hue='shot_made_flag', palette=['r','g'])
        plt.legend(labels=['miss', 'make'])
        plot_path = f'plot_learning_{figure_name}.png'
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)