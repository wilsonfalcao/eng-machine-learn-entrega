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
from kedro_datasets.pandas import ParquetDataset

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

# Salve o conjunto de dados transformado em um diretório
#data_transformed.to_parquet("data/raw/dataset_kobe_prod.paquete", index=False)

                                                                    
def load_kobe_dataset(kobe: pd.DataFrame) -> pd.DataFrame:

    import numpy as np
    
    print("Iniciando o load dos arquivo kobe_shot.csv...")
    data_transformed = kobe
    
    #Os dados devem estar localizados em "/data/raw/dataset_kobe_dev.parquet" e "/data/raw/dataset_kobe_prod.parquet" 
    print("Salvando os dados parquet DEV e PROD")
    data_transformed.to_parquet("data/01_raw/dataset_kobe_prod.parquet", index=False)
    data_transformed.to_parquet("data/01_raw/dataset_kobe_dev.parquet", index=False)

    print("Transformando os dados...")
    
    # Observe que há dados faltantes na base de dados! As linhas que possuem dados faltantes devem ser desconsideradas. Para esse exercício serão apenas consideradas as colunas
    
    data_transformed['action_type'] = pd.Categorical(data_transformed['action_type']).codes
    data_transformed['combined_shot_type'] = pd.Categorical(data_transformed['combined_shot_type']).codes
    data_transformed['shot_type'] = pd.Categorical(data_transformed['shot_type']).codes
    data_transformed['shot_zone_basic'] = pd.Categorical(data_transformed['shot_zone_basic']).codes
    data_transformed['shot_zone_area'] = pd.Categorical(data_transformed['shot_zone_area']).codes
    data_transformed['shot_zone_range'] = pd.Categorical(data_transformed['shot_zone_range']).codes
    data_transformed['team_name'] = pd.Categorical(data_transformed['team_name']).codes
    data_transformed['matchup'] = pd.Categorical(data_transformed['matchup']).codes
    data_transformed['opponent'] = pd.Categorical(data_transformed['opponent']).codes
    columns_to_remove = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_zone_range','game_date']
    data_transformed = data_transformed.drop(columns=columns_to_remove)

    # Mapear cada valor único para um número inteiro
    mapping = {valor: indice + 1 for indice, valor in enumerate(data_transformed['season'].unique())}
    
    # Criar uma nova coluna com os valores mapeados
    data_transformed['season'] = data_transformed['season'].map(mapping)
    
    
    data_transformed= data_transformed.dropna(subset=['shot_made_flag'])

    # O dataset resultante será armazenado na pasta "/data/processed/data_filtered.parquet". Ainda sobre essa seleção, qual a dimensão resultante do dataset?
    data_transformed.to_parquet("data/02_intermediate/data_filtered.parquet", index=False)
    print(data_transformed.shape)
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
        kobe_histogram(data)

def kobe_histogram(kobe_shot):

    # Limpar a figura atual
    plt.clf()

    plot_path = 'data/08_reporting/histogram_kobe_shot.png'
    
    # Separar dados por classe
    data_acerto = kobe_shot[kobe_shot['shot_made_flag'] == 1]
    data_erro = kobe_shot[kobe_shot['shot_made_flag'] == 0]
    
    # Analisar variáveis relevantes para cada classe
    
    # Acertou o arremesso
    plt.figure(figsize=(10, 6))
    
    # loc_x
    plt.subplot(1, 2, 1)
    plt.hist(data_acerto['loc_x'], bins=20, alpha=0.7, label='Acertou')
    plt.hist(data_erro['loc_x'], bins=20, alpha=0.7, label='Errou')
    plt.legend()
    plt.title('Distribuição de loc_x por Classe')
    
    # shot_distance
    plt.subplot(1, 2, 2)
    plt.hist(data_acerto['shot_distance'], bins=20, alpha=0.7, label='Acertou')
    plt.hist(data_erro['shot_distance'], bins=20, alpha=0.7, label='Errou')
    plt.legend()
    plt.title('Distribuição de shot_distance por Classe')
    
    plt.suptitle('Distribuição de Variáveis Relevantes por Classe')
    plt.show()
    
    # Errou o arremesso
    plt.figure(figsize=(10, 6))
    
    # loc_y
    plt.subplot(1, 2, 1)
    plt.hist(data_acerto['loc_y'], bins=20, alpha=0.7, label='Acertou')
    plt.hist(data_erro['loc_y'], bins=20, alpha=0.7, label='Errou')
    plt.legend()
    plt.title('Distribuição de loc_y por Classe')
    
    plt.suptitle('Distribuição de Variáveis Relevantes por Classe')

    plt.savefig(plot_path)
    plt.show()
    mlflow.log_artifact(plot_path)

def train_kobe_dataset(dataset_kobe_prod: pd.DataFrame) -> pd.DataFrame:

    # Separe os dados em treino (80%) e teste (20 %) usando uma escolha aleatória e estratificada. Armazene os datasets resultantes em "/Data/processed/base_{train|test}.parquet . Explique como a escolha de treino e teste afetam o resultado do modelo final. Quais estratégias ajudam a minimizar os efeitos de viés de dados

    # Definir as características (X) e o alvo (y)
    X = dataset_kobe_prod.drop(columns=['shot_made_flag'])
    y = dataset_kobe_prod['shot_made_flag']
    
    # Separar os dados em treino e teste (80% treino, 20% teste) com divisão estratificada
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Exibir as formas dos conjuntos de treino e teste
    print("Forma do conjunto de treino (X):", X_train.shape)
    print("Forma do conjunto de teste (X):", X_test.shape)
    print("Forma do conjunto de treino (y):", y_train.shape)
    print("Forma do conjunto de teste (y):", y_test.shape)

    # Salvar o conjunto de treino em um arquivo .parquet
    train_data = pd.concat([X_train, y_train], axis=1)
    dataset_kobe_prod.to_parquet("data/04_feature/base_train.parquet", index=False)

    # Salvando dados de teste
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_parquet("data/04_feature/base_test.parquet", index=False)

    # Plotando gráfico
    histogram_graph_train_test(y_train, y_test)

    return train_test_split

def histogram_graph_train_test(train, test):
    
    # Limpar a figura atual
    plt.clf()

    plot_path = 'data/08_reporting/histogram_train_test.png'
    
    variavel_mais_relevante = 'shot_made_flag'
    # Plotar histograma/densidade para a variável mais relevante nos conjuntos de treinamento e teste
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(train, bins=30, alpha=0.5, color='blue', label='Treino', density=True)
    plt.title('Distribuição da Variável Mais Relevante (Treino)')
    plt.xlabel('Valor da Variável')
    plt.ylabel('Densidade')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(test, bins=30, alpha=0.5, color='red', label='Teste', density=True)
    plt.title('Distribuição da Variável Mais Relevante (Teste)')
    plt.xlabel('Valor da Variável')
    plt.ylabel('Densidade')
    plt.legend()
    
    plt.tight_layout()
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

    # Salvar o melhor modelo e obtendo métricas
    model_file = "pycaret_model_1.pkl"
    model_path = "data/07_model_output/pycaret_model_1"
    save_model(final_model, model_path)
    mlflow.log_artifact("data/07_model_output/pycaret_model_1.pkl")
    main_metrics(kobe_shot, "", final_model,"Melhor Modelo")
    
    # Treinar o modelo de regressão logística
    model_file_reg = "pycaret_model_reg.pkl"
    model_path_reg = "data/07_model_output/pycaret_model_reg"
    # Criar e avaliar o modelo de Regressão Logística com validação cruzada
    lr_model = create_model('lr', cross_validation=True)

    # Salvando modelo e obtendo métricas
    save_model(lr_model, model_path_reg)
    mlflow.log_artifact("data/07_model_output/pycaret_model_reg.pkl")
    main_metrics(kobe_shot, "", lr_model,"Regressão")

    # Salvando modelo e obtendo métricas de arvore de decisão
    model_file_reg = "pycaret_model_tree.pkl"
    model_path_reg = "data/07_model_output/pycaret_model_tree"
    model_arvore = create_model('dt')

    # Salvando modelo e obtendo métricas
    save_model(model_arvore, model_path_reg)
    mlflow.log_artifact("data/07_model_output/pycaret_model_reg.pkl")
    main_metrics(kobe_shot, "", model_arvore,"Arvore")
    
    return final_model

# Funções de Métricas

def eval_metrics(pred):
    actual = pred['shot_made_flag']
    pred = pred['prediction_label']
    return (metrics.precision_score(actual, pred), 
            metrics.recall_score(actual, pred),
            metrics.f1_score(actual, pred))


def main_metrics (dataset, trained, model, name):

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
    mlflow.log_param("criterion "+name, criterion)
    mlflow.log_param("max_depth "+name, max_depth)
    
    # LOG DE METRICAS GLOBAIS
    mlflow.log_metric("precision "+name, precision)
    mlflow.log_metric("recall "+name, recall)
    mlflow.log_metric("f1  "+name, f1)
    mlflow.log_metric("Verdadeiro Positivo "+name,cm[1,1])
    mlflow.log_metric("Verdadeiro Negativo  "+name,cm[0,0])
    mlflow.log_metric("Falso Positivo  "+name,cm[0,1])
    mlflow.log_metric("Falso Negativo  "+name,cm[1,0])

    # Criar um dicionário com as métricas
    metrics_dict = {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    
    # Converter o dicionário em um DataFrame
    metrics_df = pd.DataFrame([metrics_dict])
    
    # Salvar o DataFrame em um arquivo CSV
    metrics_df.to_csv("metrics_"+name+".csv", index=False)
