import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def get_feature_pipeline(categorical_cols: list) -> ColumnTransformer:
    """
    Cria um pipeline de transformação para as variáveis (Feature Engineering).
    Utiliza o OneHotEncoder para garantir robustez em produção.
    """
    # Pipeline para variáveis categóricas
    # handle_unknown='ignore' garante que se uma categoria nova (não vista no treino)
    # aparecer em produção, o código não vai quebrar (ela será preenchida com zeros).
    # drop='first' faz o mesmo papel do drop_first=True do Pandas.
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])

    # O ColumnTransformer aplica as transformações apenas nas colunas especificadas,
    # deixando as variáveis numéricas (remainder='passthrough') intactas (sem modificação por enquanto).
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

def split_features_target(df: pd.DataFrame, target_col: str):
    """
    Separa o DataFrame entre variáveis preditoras (X) e a variável alvo (y).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int) # Garantindo que seja numérico (0/1)
    
    return X, y

# --- TESTE DAS FEATURES ---
if __name__ == "__main__":
    # Caminho do dado curado da Fase 1
    CLEAN_DATA_PATH = 'data/curated/Customer-Churn_iv_removed.csv'
    
    try:
        df_clean = pd.read_csv(CLEAN_DATA_PATH, sep=';')
        
        # Mapeando nossa variável alvo
        target = 'Churn'
        
        # Separando X e y
        X, y = split_features_target(df_clean, target)
        
        # Identificando colunas categóricas automaticamente (colunas do tipo object)
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Construindo o transformador
        preprocessor = get_feature_pipeline(categorical_cols)
        
        # Ajustando (fit) e transformando (transform) os dados
        X_transformed = preprocessor.fit_transform(X)
        
        # Recuperando o nome das colunas após o One-Hot Encoding
        cat_features_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
        numeric_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
        all_feature_names = list(cat_features_names) + numeric_cols
        
        # Criando o DataFrame final para visualização
        df_final = pd.DataFrame(X_transformed, columns=all_feature_names)
        
        print("Feature Engineering concluída com sucesso usando Scikit-Learn!")
        print(f"Dimensão original de X: {X.shape}")
        print(f"Dimensão após Encoding: {df_final.shape}")
        print("\nAlgumas colunas geradas:")
        print(df_final.columns.tolist()[:5])
        
    except FileNotFoundError:
        print("Arquivo curado não encontrado. Rode o preprocessing.py primeiro!")