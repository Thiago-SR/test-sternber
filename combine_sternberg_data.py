import pandas as pd
import os
import glob
from pathlib import Path
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_user_id(filename):
    """
    Extrai o ID do usuário do nome do arquivo.
    
    Args:
        filename: Nome do arquivo (ex: T0_4567_sternberg.csv)
        
    Returns:
        ID do usuário (ex: 4567)
    """
    # Remove a extensão .csv
    name_without_ext = filename.replace('.csv', '')
    
    # Divide pelo '_' e pega a segunda parte (índice 1)
    parts = name_without_ext.split('_')
    if len(parts) >= 2:
        return parts[1]
    else:
        return None

def extract_test_number(filename):
    """
    Extrai o número do teste do nome do arquivo.
    
    Args:
        filename: Nome do arquivo (ex: T0_4567_sternberg.csv)
        
    Returns:
        Número do teste (ex: 0, 1, 2)
    """
    # Remove a extensão .csv
    name_without_ext = filename.replace('.csv', '')
    
    # Divide pelo '_' e pega a primeira parte (índice 0)
    parts = name_without_ext.split('_')
    if len(parts) >= 1:
        # Remove o 'T' e retorna o número
        return parts[0].replace('T', '')
    else:
        return None

def convert_numeric_columns_to_int(df):
    """
    Converte colunas numéricas de float para int.
    
    Args:
        df: DataFrame
        
    Returns:
        DataFrame com colunas numéricas convertidas para int
    """
    for col in df.columns:
        # Verifica se a coluna é numérica
        if df[col].dtype in ['float64', 'float32']:
            # Verifica se todos os valores são inteiros (ou NaN)
            if df[col].dropna().apply(lambda x: x.is_integer() if pd.notna(x) else True).all():
                df[col] = df[col].astype('Int64')  # Usa Int64 para suportar NaN
                logging.info(f"  Coluna {col} convertida para int")
    
    return df

def add_prefix_to_columns(df, prefix):
    """
    Adiciona prefixo às colunas do DataFrame.
    
    Args:
        df: DataFrame
        prefix: Prefixo a ser adicionado (ex: T0_, T1_, T2_)
        
    Returns:
        DataFrame com colunas renomeadas
    """
    # Adiciona prefixo a todas as colunas
    new_columns = {col: f"{prefix}{col}" for col in df.columns}
    
    df_renamed = df.rename(columns=new_columns)
    return df_renamed

def combine_user_files(user_id, files_dict):
    """
    Combina os arquivos de um usuário específico.
    
    Args:
        user_id: ID do usuário
        files_dict: Dicionário com os arquivos organizados por teste
        
    Returns:
        DataFrame combinado
    """
    logging.info(f"Combinando arquivos para usuário {user_id}")
    
    # Lista para armazenar os DataFrames
    dfs = []
    
    # Processa cada teste (T0, T1, T2)
    for test_num in ['0', '1', '2']:
        if test_num in files_dict:
            file_path = files_dict[test_num]
            logging.info(f"  Processando {file_path}")
            
            # Lê o arquivo
            df = pd.read_csv(file_path)
            
            # Converte colunas numéricas para int
            df = convert_numeric_columns_to_int(df)
            
            # Adiciona prefixo às colunas
            prefix = f"T{test_num}_"
            df_renamed = add_prefix_to_columns(df, prefix)
            
            # Adiciona à lista
            dfs.append(df_renamed)
        else:
            logging.warning(f"  Arquivo para teste T{test_num} não encontrado para usuário {user_id}")
    
    if not dfs:
        logging.error(f"Nenhum arquivo encontrado para usuário {user_id}")
        return None
    
    # Combina os DataFrames horizontalmente
    if len(dfs) == 1:
        combined_df = dfs[0]
    else:
        # Usa o primeiro DataFrame como base e adiciona as colunas dos outros
        combined_df = dfs[0].copy()
        
        for i, df in enumerate(dfs[1:], 1):
            # Adiciona todas as colunas do DataFrame atual
            for col in df.columns:
                if col not in combined_df.columns:
                    combined_df[col] = df[col]
    
    # Converte colunas numéricas finais para int
    combined_df = convert_numeric_columns_to_int(combined_df)
    
    logging.info(f"  Arquivo combinado criado com {len(combined_df.columns)} colunas")
    return combined_df

def get_column_descriptions(columns):
    """
    Retorna uma lista de descrições curtas para cada coluna do teste Sternberg.
    """
    desc_map = {
        'subNum': 'ID',
        'length': 'Tamanho',
        'trial': 'Trial',
        'set': 'Conjunto',
        'stim': 'Estímulo',
        'targetfoil': 'Verdadeiro/Falso',
        'resp': 'Resposta',
        'corr': 'Correto',
        'rt': 'Tempo(ms)',
    }
    # Para colunas com prefixo (T0_, T1_, T2_, etc)
    def desc(col):
        for prefix in ['T0_', 'T1_', 'T2_']:
            if col.startswith(prefix):
                base = col[len(prefix):]
                return desc_map.get(base, base)
        return desc_map.get(col, col)
    return [desc(col) for col in columns]

def main():
    # Criar pasta para os arquivos combinados
    output_folder = "dados_sternberg_combinados"
    Path(output_folder).mkdir(exist_ok=True)
    
    # Pasta com os arquivos de dados
    input_folder = "data"
    
    # Verificar se a pasta existe
    if not os.path.exists(input_folder):
        logging.error(f"Pasta {input_folder} não encontrada!")
        return
    
    # Encontrar todos os arquivos CSV
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        logging.error(f"Nenhum arquivo CSV encontrado em {input_folder}")
        return
    
    logging.info(f"Encontrados {len(csv_files)} arquivos CSV")
    
    # Organizar arquivos por usuário
    users_files = {}
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        user_id = extract_user_id(filename)
        test_num = extract_test_number(filename)
        
        if user_id and test_num:
            if user_id not in users_files:
                users_files[user_id] = {}
            
            users_files[user_id][test_num] = file_path
            logging.info(f"Arquivo {filename} -> Usuário {user_id}, Teste T{test_num}")
    
    logging.info(f"Organizados {len(users_files)} usuários")
    
    # Processar cada usuário
    for user_id, files_dict in users_files.items():
        logging.info(f"Processando usuário {user_id}")
        
        # Verificar se tem todos os 3 testes
        if len(files_dict) < 3:
            logging.warning(f"Usuário {user_id} tem apenas {len(files_dict)} testes (esperado: 3)")
        
        # Combinar arquivos do usuário
        combined_df = combine_user_files(user_id, files_dict)
        
        if combined_df is not None:
            # Salvar arquivo combinado
            output_file = os.path.join(output_folder, f"{user_id}_sternberg_combined.csv")
            # Adicionar linha de descrição curta acima do cabeçalho
            descriptions = get_column_descriptions(combined_df.columns)
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                f.write(','.join(descriptions) + '\n')
                # Salva o DataFrame garantindo que colunas numéricas sejam salvas como números
                combined_df.to_csv(f, index=False, float_format='%.0f')
            logging.info(f"Arquivo salvo: {output_file}")
            
            # Mostrar informações sobre as colunas
            logging.info(f"  Colunas: {list(combined_df.columns)}")
            logging.info(f"  Linhas: {len(combined_df)}")
        else:
            logging.error(f"Erro ao combinar arquivos para usuário {user_id}")
    
    logging.info("Processamento concluído!")

if __name__ == "__main__":
    main() 