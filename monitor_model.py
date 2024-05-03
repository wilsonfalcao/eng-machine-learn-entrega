import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Carregar o modelo treinado
model_path = 'data/07_model_output/pycaret_model_1.pkl'
model = joblib.load(model_path)

# Título da aplicação
st.title('Aplicação de Previsão de Arremessos do Kobe Bryant')

# Subtítulo
st.write('Esta é uma aplicação simples para fazer previsões sobre se um arremesso do Kobe Bryant será bem-sucedido ou não.')

# Carregar os dados do Kobe Shot
@st.cache
def load_data():
    return pd.read_csv('data/01_raw/dataset_kobe_prod.parquet')

# Carregar os dados
df = load_data()

# Exibir os dados (opcional)
# st.write(df)

# Seção para fazer previsões
st.sidebar.title('Faça uma Previsão')
action_type = st.sidebar.slider('action_type', min_value=1, max_value=df['action_type'].max(), step=1, value=26)
combined_shot_type = st.sidebar.slider('combined_shot_type', min_value=1, max_value=df['combined_shot_type'].max(), step=1, value=3)
game_event_id = st.sidebar.slider('game_event_id', min_value=1, max_value=df['game_event_id'].max(), step=1, value=12)
game_id = st.sidebar.slider('game_id', min_value=20000000, max_value=49900000, step=1, value=2000012)
loc_x = st.sidebar.slider('loc_x', min_value=-250, max_value=250, step=1, value=-157)
loc_y = st.sidebar.slider('loc_y', min_value=-50, max_value=900, step=1, value=0)
season = st.sidebar.slider('season', min_value=1996, max_value=2016, step=1, value=1)
seconds_remaining = st.sidebar.slider('seconds_remaining', min_value=0, max_value=720, step=1, value=22)
shot_distance = st.sidebar.slider('shot_distance', min_value=0, max_value=60, step=1, value=15)
shot_type = st.sidebar.slider('shot_type', min_value=0, max_value=df['shot_type'].max(), value=0)
shot_zone_area = st.sidebar.slider('shot_zone_area', min_value=0, max_value=df['shot_zone_area'].max(), value=3)
shot_zone_basic = st.sidebar.slider('shot_zone_basic', min_value=0, max_value=df['shot_zone_basic'].max(), value=4)
team_id = st.sidebar.slider('team_id', min_value=0, max_value=1610612766, step=1, value=0)
team_name = st.sidebar.slider('team_name', min_value=0, max_value=df['team_name'].max(), value=1)
matchup = st.sidebar.slider('matchup', min_value=0, max_value=df['matchup'].max(), value=28)
opponent = st.sidebar.slider('opponent', min_value=0, max_value=df['opponent'].max(), value=25)
shot_id = st.sidebar.slider('shot_id', min_value=0, max_value=df['shot_id'].max(), step=1, value=2)


# Fazer a previsão
if st.sidebar.button('Prever'):

    data_values = {
        'action_type':action_type,
        'combined_shot_type':combined_shot_type,
        'game_event_id': game_event_id,
        'game_id': game_id,
        'loc_x': loc_x,
        'loc_y': loc_y,
        'season': season,
        'seconds_remaining':seconds_remaining,
        'shot_distance': shot_distance,
        'shot_type': shot_type,
        'shot_zone_area':shot_zone_area,
        'shot_zone_basic': shot_zone_basic,
        'team_id': team_id,
        'team_name': team_name,
        'matchup':matchup,
        'opponent': opponent,
        'shot_id': shot_id
    }
    
    # Converter os dados para DataFrame
    df_predict = pd.DataFrame(data_values, index=[0])

    prediction = model.predict(df_predict)
    if prediction[0] == 1:
        st.sidebar.success('O arremesso foi bem-sucedido!')
    else:
        st.sidebar.error('O arremesso não foi bem-sucedido.')

# Rodapé
st.write('')
st.write('Desenvolvido por Wilson Falcão')