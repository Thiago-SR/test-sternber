import pandas as pd
import os
import glob
import numpy as np

def process_rt_means():
    """
    Processa todos os arquivos CSV na pasta dados_sternberg_combinados,
    calcula as médias das colunas T0_rt, T1_rt e T2_rt,
    calcula as médias por length para cada teste,
    calcula as médias para respostas corretas (corr = 1),
    calcula as médias para respostas incorretas (corr = 0),
    calcula a precisão (accuracy) para cada teste,
    e salva os resultados em um único arquivo.
    """
    
    # Caminho para a pasta com os dados
    data_folder = "dados_sternberg_combinados"
    
    # Lista para armazenar os resultados
    results = []
    
    # Encontrar todos os arquivos CSV na pasta
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    
    print(f"Encontrados {len(csv_files)} arquivos CSV para processar...")
    
    for file_path in csv_files:
        try:
            # Extrair o nome do arquivo (sem extensão) para usar como identificador
            file_name = os.path.basename(file_path)
            participant_id = file_name.replace("_sternberg_combined.csv", "")
            
            print(f"Processando arquivo: {file_name}")
            
            # Ler o arquivo CSV, pulando a primeira linha (cabeçalho)
            df = pd.read_csv(file_path, skiprows=1)
            
            # Verificar se as colunas necessárias existem
            required_columns = ['T0_rt', 'T1_rt', 'T2_rt']
            length_columns = ['T0_length', 'T1_length', 'T2_length']
            corr_columns = ['T0_corr', 'T1_corr', 'T2_corr']
            missing_columns = [col for col in required_columns + length_columns + corr_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Aviso: Colunas ausentes no arquivo {file_name}: {missing_columns}")
                continue
            
            # Converter colunas rt para numérico, tratando valores inválidos
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Converter colunas length para numérico
            for col in length_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Converter colunas corr para numérico
            for col in corr_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remover valores NaN
            valid_data = {}
            length_data = {}
            correct_data = {}
            incorrect_data = {}
            accuracy_data = {}
            accuracy_by_length_data = {}
            
            for i, col in enumerate(required_columns):
                # Filtrar valores válidos (remover apenas NaN)
                valid_values = df[col].dropna()
                
                if len(valid_values) == 0:
                    print(f"  Aviso: Nenhum valor válido encontrado para {col}")
                    valid_data[col] = np.nan
                else:
                    valid_data[col] = valid_values.mean()
                    print(f"  - {col}: {valid_data[col]:.3f} (n={len(valid_values)} valores válidos)")
                
                # Calcular médias por length para este teste
                length_col = length_columns[i]
                test_prefix = col.replace('_rt', '')  # T0, T1, T2
                
                # Criar DataFrame temporário com rt e length válidos
                temp_df = df[[col, length_col]].copy()
                temp_df = temp_df.dropna()
                
                if len(temp_df) > 0:
                    # Agrupar por length e calcular média
                    length_means = temp_df.groupby(length_col)[col].mean()
                    
                    # Adicionar ao dicionário de resultados por length
                    for length_val, mean_rt in length_means.items():
                        key = f"mean_rt_by_length_{test_prefix}_{int(length_val)}"
                        length_data[key] = mean_rt
                        print(f"    - {key}: {mean_rt:.3f}")
                else:
                    print(f"    - Aviso: Nenhum valor válido encontrado para {test_prefix} por length")
                
                # Calcular médias para respostas corretas (corr = 1)
                corr_col = corr_columns[i]
                
                # Criar DataFrame temporário com rt e corr válidos
                temp_corr_df = df[[col, corr_col]].copy()
                temp_corr_df = temp_corr_df.dropna()
                
                # Filtrar apenas respostas corretas (corr = 1)
                correct_responses = temp_corr_df[temp_corr_df[corr_col] == 1]
                
                if len(correct_responses) > 0:
                    correct_mean = correct_responses[col].mean()
                    key = f"mean_rt_correct_{test_prefix}"
                    correct_data[key] = correct_mean
                    print(f"    - {key}: {correct_mean:.3f} (n={len(correct_responses)} respostas corretas)")
                else:
                    print(f"    - Aviso: Nenhuma resposta correta encontrada para {test_prefix}")
                
                # Calcular médias para respostas incorretas (corr = 0)
                incorrect_responses = temp_corr_df[temp_corr_df[corr_col] == 0]
                
                if len(incorrect_responses) > 0:
                    incorrect_mean = incorrect_responses[col].mean()
                    key = f"mean_rt_incorrect_{test_prefix}"
                    incorrect_data[key] = incorrect_mean
                    print(f"    - {key}: {incorrect_mean:.3f} (n={len(incorrect_responses)} respostas incorretas)")
                else:
                    print(f"    - Aviso: Nenhuma resposta incorreta encontrada para {test_prefix}")
                
                # Calcular accuracy (proporção de trials com corr = 1)
                if len(temp_corr_df) > 0:
                    accuracy = len(correct_responses) / len(temp_corr_df)
                    key = f"accuracy_total_{test_prefix}"
                    accuracy_data[key] = accuracy
                    print(f"    - {key}: {accuracy:.3f} ({len(correct_responses)}/{len(temp_corr_df)} respostas corretas)")
                else:
                    print(f"    - Aviso: Nenhum valor válido encontrado para accuracy de {test_prefix}")
                
                # Calcular accuracy por length para este teste
                temp_length_corr_df = df[[length_col, corr_col]].copy()
                temp_length_corr_df = temp_length_corr_df.dropna()
                
                if len(temp_length_corr_df) > 0:
                    # Agrupar por length e calcular accuracy (proporção de corr == 1)
                    length_accuracy = temp_length_corr_df.groupby(length_col)[corr_col].apply(
                        lambda x: (x == 1).sum() / len(x)
                    )
                    
                    # Adicionar ao dicionário de resultados por length
                    for length_val, accuracy_val in length_accuracy.items():
                        key = f"accuracy_by_length_{test_prefix}_{int(length_val)}"
                        accuracy_by_length_data[key] = accuracy_val
                        print(f"    - {key}: {accuracy_val:.3f}")
                else:
                    print(f"    - Aviso: Nenhum valor válido encontrado para accuracy por length de {test_prefix}")
            
            # Adicionar resultados à lista - organizando por T0, T1, T2
            result_dict = {
                'id': participant_id
            }
            
            # Adicionar dados T0 primeiro
            result_dict['mean_rt_total_T0'] = valid_data['T0_rt']
            for key, value in length_data.items():
                if 'T0' in key:
                    result_dict[key] = value
            for key, value in correct_data.items():
                if 'T0' in key:
                    result_dict[key] = value
            for key, value in incorrect_data.items():
                if 'T0' in key:
                    result_dict[key] = value
            for key, value in accuracy_data.items():
                if 'T0' in key:
                    result_dict[key] = value
            for key, value in accuracy_by_length_data.items():
                if 'T0' in key:
                    result_dict[key] = value
            
            # Adicionar dados T1
            result_dict['mean_rt_total_T1'] = valid_data['T1_rt']
            for key, value in length_data.items():
                if 'T1' in key:
                    result_dict[key] = value
            for key, value in correct_data.items():
                if 'T1' in key:
                    result_dict[key] = value
            for key, value in incorrect_data.items():
                if 'T1' in key:
                    result_dict[key] = value
            for key, value in accuracy_data.items():
                if 'T1' in key:
                    result_dict[key] = value
            for key, value in accuracy_by_length_data.items():
                if 'T1' in key:
                    result_dict[key] = value
            
            # Adicionar dados T2
            result_dict['mean_rt_total_T2'] = valid_data['T2_rt']
            for key, value in length_data.items():
                if 'T2' in key:
                    result_dict[key] = value
            for key, value in correct_data.items():
                if 'T2' in key:
                    result_dict[key] = value
            for key, value in incorrect_data.items():
                if 'T2' in key:
                    result_dict[key] = value
            for key, value in accuracy_data.items():
                if 'T2' in key:
                    result_dict[key] = value
            for key, value in accuracy_by_length_data.items():
                if 'T2' in key:
                    result_dict[key] = value
            
            results.append(result_dict)
            
        except Exception as e:
            print(f"Erro ao processar arquivo {file_path}: {str(e)}")
            continue
    
    # Criar DataFrame com todos os resultados
    if results:
        results_df = pd.DataFrame(results)
        
        # Salvar resultados em um arquivo CSV
        output_file = "analises.csv"
        
        # Salvar sem formatação forçada de casas decimais
        results_df.to_csv(output_file, index=False)
        
        print(f"\nProcessamento concluído!")
        print(f"Resultados salvos em: {output_file}")
        print(f"Total de participantes processados: {len(results)}")
        
        # Mostrar um resumo dos resultados
        print("\nResumo dos resultados:")
        print(results_df.describe())
        
        # Verificar se há valores NaN
        nan_counts = results_df.isna().sum()
        if nan_counts.sum() > 0:
            print(f"\nValores NaN encontrados:")
            print(nan_counts)
        
        return results_df
    else:
        print("Nenhum resultado foi gerado.")
        return None

if __name__ == "__main__":
    process_rt_means() 