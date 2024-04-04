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

# Configuracao
criterion = 'gini'
max_depth = 5


def load_kobe_dataset(kobe: pd.DataFrame) -> pd.DataFrame:

    import numpy as np
    
    print("Started function cleanup data...")

    # remove data target null or blank
    data_transformed = kobe
    data_transformed= data_transformed.dropna(subset=['shot_made_flag'])

    # Salve o conjunto de dados transformado em um diretório
    data_transformed.to_csv("data/02_intermediate/kobe_shot_transformed.csv", index=False)
    
    return data_transformed
    
def kobe_avarage_shot_artefact(kobe_shot: pd.DataFrame) -> pd.DataFrame:

        plot_path = 'data/08_reporting/shots_made_and_missed.png'
        data = kobe_shot
        sns.set(style='ticks')
            
        #percentage = [ (made shots) / (total shots) ] * 100
        made_shots_num = data['shot_made_flag'].value_counts()[1]
        total_shots = data['shot_made_flag'].value_counts()[0]+ data['shot_made_flag'].value_counts()[1]
        fg_percentage = 100 * made_shots_num / total_shots
        sns.countplot(data=data,x='shot_made_flag',hue='shot_made_flag', palette=['r','g'])
        plt.legend(labels=['miss', 'make'])
        plt.savefig(plot_path)
        plt.show()
        mlflow.log_artifact(plot_path)


def build_kobe_model_pycaret(kobe_shot: pd.DataFrame) -> pd.DataFrame:
    
    from pycaret.classification import setup, compare_models, save_model, create_model
    
    setup(kobe_shot, target='shot_made_flag', categorical_features=['action_type', 'combined_shot_type'])

    # Comparar modelos disponíveis
    best_model = compare_models()
    
    # Treinar o melhor modelo
    final_model = create_model(best_model)

    # Salvar o melhor modelo
    model_file = "pycaret_model_1.pkl"
    model_path = "data/07_model_output/pycaret_model_1"
    save_model(final_model, model_path)
    
    mlflow.log_artifact("data/07_model_output/pycaret_model_1.pkl")

    main_metrics(kobe_shot, "", final_model)

    return final_model


# Funções de Métricas

def eval_metrics(pred):
    actual = pred['shot_made_flag']
    pred = pred['prediction_label']
    return (metrics.precision_score(actual, pred), 
            metrics.recall_score(actual, pred),
            metrics.f1_score(actual, pred))


def main_metrics (dataset, trained, model):

    from pycaret.classification import predict_model
    
    # Fazer previsões usando o modelo treinado
    predictions = predict_model(model, dataset)

    (precision, recall, f1) = eval_metrics(predictions)
    cm =  metrics.confusion_matrix(predictions["shot_made_flag"], predictions['prediction_label'])

    print("Decisn Tree Classifier (criterion=%s, max_depth=%f):" % (criterion, max_depth))
    print("  precision: %s" % precision)
    print("  recall: %s" % recall)
    print("  f1: %s" % f1)

    # LOG DE PARAMETROS DO MODELO
    mlflow.log_param("criterion", criterion)
    mlflow.log_param("max_depth", max_depth)
    
    # LOG DE METRICAS GLOBAIS
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("Verdadeiro Positivo",cm[1,1])
    mlflow.log_metric("Verdadeiro Negativo",cm[0,0])
    mlflow.log_metric("Falso Positivo",cm[0,1])
    mlflow.log_metric("Falso Negativo",cm[1,0])
