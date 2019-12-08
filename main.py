import streamlit as st
import pandas as pd
pd.set_option('display.max_columns',500)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import joblib

st.title('DESAFIO SME')
dtypes = {'NIVEL':object}

@st.cache(suppress_st_warning=True)
def load_model():
    df_final = pd.read_csv('DADOS_FINAIS.csv',sep=';',dtype=dtypes, encoding = "ISO-8859-1",decimal=',')

    return df_final

df_final = load_model()

@st.cache(suppress_st_warning=True)
def pre_pro(df0):
    df0 = df0.dropna()
    df00 = df0[['CARGO', 'TURNO', 'IDADE', 
        'G1 P2', 'G1 P9',
#                'GENERO/SEXO', 
#         'ESTADO CIVIL', 
        'ESCORE', 'NIVEL',
#         'Variavel 1', 
        'Valor Variavel 1', 
#                'Variavel 2',
'Valor Variavel 2', 
#                'Variavel 3', 
        'Valor Variavel 3', 
#                'Variavel 4',
'Valor Variavel 4', 
#                'Variavel 5', 
        'Valor Variavel 5',
    'ACIDENTE_DIA']]
    df00d=pd.get_dummies(df00)

    X0 = df00d.drop('ACIDENTE_DIA',1)
    y0 = df00d.ACIDENTE_DIA
    sm = SMOTE(ratio='minority', random_state=42)
    X0_sm, y0_sm = sm.fit_resample(X0,y0)
    X_train, X_valid, y_train, y_valid = train_test_split(X0_sm, y0_sm, test_size=0.3, random_state=42)
    return X_train, X_valid, y_train, y_valid, X0_sm, y0_sm


@st.cache(suppress_st_warning=True)
def accept_user_data():
    idade = st.text_input('Colocar IDADE:')
    cargo = st.text_input('Cargo:')
    turno = st.text_input('Turno')
    g1_p2 = st.text_input('G1 P2')
    g1_p9 = st.text_input("G1 P9")
    ESCORE = st.text_input('ESCORE')
    NIVEL = st.text_input('NIVEL')
    VALOR_VARIAVEL_1 = st.text_input('VAlor Variavel 1')
    VALOR_VARIAVEL_2 = st.text_input('Valor Variavel 2')
    VALOR_VARIAVEL_3 = st.text_input('Valor Variavel 3')
    VALOR_VARIAVEL_4 = st.text_input('Valor Variavel 4')
    VALOR_VARIAVEL_5 = st.text_input('Valor Variavel 5')
    user_prediction_data = np.array([idade, cargo, turno, g1_p2, g1_p9, ESCORE, NIVEL,VALOR_VARIAVEL_1, VALOR_VARIAVEL_2,  VALOR_VARIAVEL_3, 
    VALOR_VARIAVEL_4, VALOR_VARIAVEL_5])
    return user_prediction_data

X_train, X_valid, y_train, y_valid, X0_sm, y0_sm = pre_pro(df_final)

if st.checkbox('Dados SME'):
    st.write(df_final.head())
if st.checkbox('Input Dados'):
    user_prediction_data = accept_user_data()


@st.cache(suppress_st_warning=True)
def floresta(X_train, X_valid, y_train, y_valid, X0_sm, y0_sm):
    # TREINO
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # PREDIÇÃO
    y_pred0 = rf.predict(X_valid)
    X_valid2 = X_valid.reshape(-1, 1)
    y_pred2 = rf.predict_proba(X_valid)

    # SCORE
    score0 = metrics.accuracy_score(y_valid, y_pred0) * 100
    report0 = classification_report(y_valid, y_pred0)

    return report0, score0,y_pred2

def logisticRegression(X_train, X_valid, y_train, y_valid, X0_sm, y0_sm):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred0 = lr.predict(X_valid)
    score0 = metrics.accuracy_score(y_valid, y_pred0) * 100
    report0 = classification_report(y_valid, y_pred0)
    return score0, report0

choose_model = st.sidebar.selectbox("Escolha seu modelo",
			   ["Nenhum", "Random Forest", 'XGBoost','Regressao Logistica'])

if choose_model == "Nenhum":
    st.write(" ")
    st.write("A partir dos testes da SME, agora é possível predizer Acidentes de Trabalho!")
    st.write(" ")
    st.write("Escolha seu modelo!!")

elif choose_model == "Random Forest":
    report0, score0,y_pred2 = floresta(X_train, X_valid, y_train, y_valid, X0_sm, y0_sm)
    st.text("A acurácia NO TREINO do Random Forest é: ")
    st.write(score0,"%")
    st.text("A acurácia NO TESTE do Random Forest é: ")
    st.text("O relatório da Classificação da Random Forest é: ")
    st.write(report0)
    st.text("Probabilidade de acidente: ")
    st.write(y_pred2)

elif choose_model == "Regressao Logistica":
    report1, score1  = logisticRegression(X_train, X_valid, y_train, y_valid, X0_sm, y0_sm)
    st.text("A acurácia NO TREINO do Regressão Logistica e: ")
    st.write(score1,"%")
    st.text("A acurácia NO TESTE do Regressão Logística é: ")
    st.text("O relatório da Classificação da Regressão Logística é: ")
    st.write(report1)




