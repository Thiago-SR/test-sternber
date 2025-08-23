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
            targetfoil_columns = ['T0_targetfoil', 'T1_targetfoil', 'T2_targetfoil']
            missing_columns = [col for col in required_columns + length_columns + corr_columns + targetfoil_columns if col not in df.columns]
            
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
            
            # Calcular slope do RT por length para T0, T1 e T2
            slope_data = {}
            for test_prefix in ['T0', 'T1', 'T2']:
                rt_col = f'{test_prefix}_rt'
                length_col = f'{test_prefix}_length'
                corr_col = f'{test_prefix}_corr'
                
                if rt_col in df.columns and length_col in df.columns and corr_col in df.columns:
                    # Criar DataFrame temporário com rt, length e corr válidos
                    temp_slope_df = df[[rt_col, length_col, corr_col]].copy()
                    temp_slope_df = temp_slope_df.dropna()
                    
                    if len(temp_slope_df) > 0:
                        # Filtrar apenas respostas corretas (corr = 1)
                        correct_trials = temp_slope_df[temp_slope_df[corr_col] == 1]
                        
                        if len(correct_trials) > 0:
                            # Agrupar por length e calcular RT médio
                            rt_by_length = correct_trials.groupby(length_col)[rt_col].mean().reset_index()
                            
                            if len(rt_by_length) > 1:  # Precisa de pelo menos 2 pontos para regressão
                                # Calcular regressão linear: RT_médio = β₀ + β₁ · length
                                # Usando numpy para calcular o slope
                                x = rt_by_length[length_col].values
                                y = rt_by_length[rt_col].values
                                
                                # Calcular slope usando numpy.polyfit (grau 1 = regressão linear)
                                slope, intercept = np.polyfit(x, y, 1)
                                
                                slope_data[f'slope_rt_by_length_{test_prefix}'] = slope
                                print(f"    - slope_rt_by_length_{test_prefix}: {slope:.3f} ms/item (n={len(rt_by_length)} pontos)")
                            else:
                                print(f"    - Aviso: Insuficientes pontos para calcular slope de {test_prefix} (apenas {len(rt_by_length)} ponto)")
                                slope_data[f'slope_rt_by_length_{test_prefix}'] = np.nan
                        else:
                            print(f"    - Aviso: Nenhuma resposta correta encontrada para calcular slope de {test_prefix}")
                            slope_data[f'slope_rt_by_length_{test_prefix}'] = np.nan
                    else:
                        print(f"    - Aviso: Nenhum valor válido encontrado para calcular slope de {test_prefix}")
                        slope_data[f'slope_rt_by_length_{test_prefix}'] = np.nan
                else:
                    print(f"    - Aviso: Colunas {rt_col}, {length_col} ou {corr_col} não encontradas")
                    slope_data[f'slope_rt_by_length_{test_prefix}'] = np.nan
            
            # Calcular RT médio por acerto por length para T0, T1 e T2
            rt_correct_by_length_data = {}
            for test_prefix in ['T0', 'T1', 'T2']:
                rt_col = f'{test_prefix}_rt'
                length_col = f'{test_prefix}_length'
                corr_col = f'{test_prefix}_corr'
                
                if rt_col in df.columns and length_col in df.columns and corr_col in df.columns:
                    # Criar DataFrame temporário com rt, length e corr válidos
                    temp_rt_correct_df = df[[rt_col, length_col, corr_col]].copy()
                    temp_rt_correct_df = temp_rt_correct_df.dropna()
                    
                    if len(temp_rt_correct_df) > 0:
                        # Filtrar apenas respostas corretas (corr = 1)
                        correct_trials = temp_rt_correct_df[temp_rt_correct_df[corr_col] == 1]
                        
                        if len(correct_trials) > 0:
                            # Agrupar por length e calcular RT médio para respostas corretas
                            rt_correct_by_length = correct_trials.groupby(length_col)[rt_col].mean()
                            
                            # Adicionar ao dicionário de resultados por length
                            for length_val, mean_rt in rt_correct_by_length.items():
                                key = f"mean_rt_correct_by_length_{test_prefix}_{int(length_val)}"
                                rt_correct_by_length_data[key] = mean_rt
                                print(f"    - {key}: {mean_rt:.3f} ms")
                        else:
                            print(f"    - Aviso: Nenhuma resposta correta encontrada para calcular RT por length de {test_prefix}")
                            # Adicionar valores NaN para todos os lengths
                            for length_val in [2, 4, 6]:
                                key = f"mean_rt_correct_by_length_{test_prefix}_{length_val}"
                                rt_correct_by_length_data[key] = np.nan
                    else:
                        print(f"    - Aviso: Nenhum valor válido encontrado para calcular RT por length de {test_prefix}")
                        # Adicionar valores NaN para todos os lengths
                        for length_val in [2, 4, 6]:
                            key = f"mean_rt_correct_by_length_{test_prefix}_{length_val}"
                            rt_correct_by_length_data[key] = np.nan
                else:
                    print(f"    - Aviso: Colunas {rt_col}, {length_col} ou {corr_col} não encontradas")
                    # Adicionar valores NaN para todos os lengths
                    for length_val in [2, 4, 6]:
                        key = f"mean_rt_correct_by_length_{test_prefix}_{length_val}"
                        rt_correct_by_length_data[key] = np.nan
            
            # Calcular RT médio por erro por length para T0, T1 e T2
            rt_incorrect_by_length_data = {}
            for test_prefix in ['T0', 'T1', 'T2']:
                rt_col = f'{test_prefix}_rt'
                length_col = f'{test_prefix}_length'
                corr_col = f'{test_prefix}_corr'
                
                if rt_col in df.columns and length_col in df.columns and corr_col in df.columns:
                    # Criar DataFrame temporário com rt, length e corr válidos
                    temp_rt_incorrect_df = df[[rt_col, length_col, corr_col]].copy()
                    temp_rt_incorrect_df = temp_rt_incorrect_df.dropna()
                    
                    if len(temp_rt_incorrect_df) > 0:
                        # Filtrar apenas respostas incorretas (corr = 0)
                        incorrect_trials = temp_rt_incorrect_df[temp_rt_incorrect_df[corr_col] == 0]
                        
                        if len(incorrect_trials) > 0:
                            # Agrupar por length e calcular RT médio para respostas incorretas
                            rt_incorrect_by_length = incorrect_trials.groupby(length_col)[rt_col].mean()
                            
                            # Adicionar ao dicionário de resultados por length
                            for length_val, mean_rt in rt_incorrect_by_length.items():
                                key = f"mean_rt_incorrect_by_length_{test_prefix}_{int(length_val)}"
                                rt_incorrect_by_length_data[key] = mean_rt
                                print(f"    - {key}: {mean_rt:.3f} ms")
                        else:
                            print(f"    - Aviso: Nenhuma resposta incorreta encontrada para calcular RT por length de {test_prefix}")
                            # Adicionar valores NaN para todos os lengths
                            for length_val in [2, 4, 6]:
                                key = f"mean_rt_incorrect_by_length_{test_prefix}_{length_val}"
                                rt_incorrect_by_length_data[key] = np.nan
                    else:
                        print(f"    - Aviso: Nenhum valor válido encontrado para calcular RT por length de {test_prefix}")
                        # Adicionar valores NaN para todos os lengths
                        for length_val in [2, 4, 6]:
                            key = f"mean_rt_incorrect_by_length_{test_prefix}_{length_val}"
                            rt_incorrect_by_length_data[key] = np.nan
                else:
                    print(f"    - Aviso: Colunas {rt_col}, {length_col} ou {corr_col} não encontradas")
                    # Adicionar valores NaN para todos os lengths
                    for length_val in [2, 4, 6]:
                        key = f"mean_rt_incorrect_by_length_{test_prefix}_{length_val}"
                        rt_incorrect_by_length_data[key] = np.nan
            
            # Calcular acurácia para alvos (T) e foils (F) por length para T0, T1 e T2
            targetfoil_accuracy_by_length_data = {}
            for test_prefix in ['T0', 'T1', 'T2']:
                length_col = f'{test_prefix}_length'
                targetfoil_col = f'{test_prefix}_targetfoil'
                corr_col = f'{test_prefix}_corr'
                
                if length_col in df.columns and targetfoil_col in df.columns and corr_col in df.columns:
                    # Criar DataFrame temporário com length, targetfoil e corr válidos
                    temp_targetfoil_length_df = df[[length_col, targetfoil_col, corr_col]].copy()
                    temp_targetfoil_length_df = temp_targetfoil_length_df.dropna()
                    
                    if len(temp_targetfoil_length_df) > 0:
                        # Agrupar por length e targetfoil e calcular proporções de corr == 1
                        for length_val in [2, 4, 6]:
                            # Filtrar por length específico
                            length_trials = temp_targetfoil_length_df[temp_targetfoil_length_df[length_col] == length_val]
                            
                            if len(length_trials) > 0:
                                # Filtrar por target (T) e foil (F)
                                target_trials = length_trials[length_trials[targetfoil_col] == 'T']
                                foil_trials = length_trials[length_trials[targetfoil_col] == 'F']
                                
                                # Calcular accuracy para target trials deste length
                                if len(target_trials) > 0:
                                    target_accuracy = (target_trials[corr_col] == 1).sum() / len(target_trials)
                                    key = f"accuracy_target_by_length_{test_prefix}_{int(length_val)}"
                                    targetfoil_accuracy_by_length_data[key] = target_accuracy
                                    print(f"    - {key}: {target_accuracy:.3f} ({len(target_trials)} trials target)")
                                else:
                                    key = f"accuracy_target_by_length_{test_prefix}_{int(length_val)}"
                                    targetfoil_accuracy_by_length_data[key] = np.nan
                                    print(f"    - {key}: NaN (sem trials target)")
                                
                                # Calcular accuracy para foil trials deste length
                                if len(foil_trials) > 0:
                                    foil_accuracy = (foil_trials[corr_col] == 1).sum() / len(foil_trials)
                                    key = f"accuracy_foil_by_length_{test_prefix}_{int(length_val)}"
                                    targetfoil_accuracy_by_length_data[key] = foil_accuracy
                                    print(f"    - {key}: {foil_accuracy:.3f} ({len(foil_trials)} trials foil)")
                                else:
                                    key = f"accuracy_foil_by_length_{test_prefix}_{int(length_val)}"
                                    targetfoil_accuracy_by_length_data[key] = np.nan
                                    print(f"    - {key}: NaN (sem trials foil)")
                            else:
                                # Adicionar valores NaN para este length se não houver trials
                                key_target = f"accuracy_target_by_length_{test_prefix}_{int(length_val)}"
                                key_foil = f"accuracy_foil_by_length_{test_prefix}_{int(length_val)}"
                                targetfoil_accuracy_by_length_data[key_target] = np.nan
                                targetfoil_accuracy_by_length_data[key_foil] = np.nan
                                print(f"    - {key_target}: NaN (sem trials para length {length_val})")
                                print(f"    - {key_foil}: NaN (sem trials para length {length_val})")
                    else:
                        print(f"    - Aviso: Nenhum valor válido encontrado para targetfoil accuracy por length de {test_prefix}")
                        # Adicionar valores NaN para todos os lengths
                        for length_val in [2, 4, 6]:
                            key_target = f"accuracy_target_by_length_{test_prefix}_{int(length_val)}"
                            key_foil = f"accuracy_foil_by_length_{test_prefix}_{int(length_val)}"
                            targetfoil_accuracy_by_length_data[key_target] = np.nan
                            targetfoil_accuracy_by_length_data[key_foil] = np.nan
                else:
                    print(f"    - Aviso: Colunas {length_col}, {targetfoil_col} ou {corr_col} não encontradas")
                    # Adicionar valores NaN para todos os lengths
                    for length_val in [2, 4, 6]:
                        key_target = f"accuracy_target_by_length_{test_prefix}_{int(length_val)}"
                        key_foil = f"accuracy_foil_by_length_{test_prefix}_{int(length_val)}"
                        targetfoil_accuracy_by_length_data[key_target] = np.nan
                        targetfoil_accuracy_by_length_data[key_foil] = np.nan
            
            # Calcular accuracy para target vs foil para T0, T1 e T2
            targetfoil_accuracy_data = {}
            for test_prefix in ['T0', 'T1', 'T2']:
                targetfoil_col = f'{test_prefix}_targetfoil'
                corr_col = f'{test_prefix}_corr'
                
                if targetfoil_col in df.columns and corr_col in df.columns:
                    # Criar DataFrame temporário com targetfoil e corr válidos
                    temp_targetfoil_df = df[[targetfoil_col, corr_col]].copy()
                    temp_targetfoil_df = temp_targetfoil_df.dropna()
                    
                    if len(temp_targetfoil_df) > 0:
                        # Filtrar por target (T) e foil (F)
                        target_trials = temp_targetfoil_df[temp_targetfoil_df[targetfoil_col] == 'T']
                        foil_trials = temp_targetfoil_df[temp_targetfoil_df[targetfoil_col] == 'F']
                        
                        # Calcular accuracy para target trials
                        if len(target_trials) > 0:
                            target_accuracy = (target_trials[corr_col] == 1).sum() / len(target_trials)
                            targetfoil_accuracy_data[f'accuracy_target_{test_prefix}'] = target_accuracy
                            print(f"    - accuracy_target_{test_prefix}: {target_accuracy:.3f} ({len(target_trials)} trials target)")
                        else:
                            print(f"    - Aviso: Nenhum trial target encontrado para {test_prefix}")
                            targetfoil_accuracy_data[f'accuracy_target_{test_prefix}'] = np.nan
                        
                        # Calcular accuracy para foil trials
                        if len(foil_trials) > 0:
                            foil_accuracy = (foil_trials[corr_col] == 1).sum() / len(foil_trials)
                            targetfoil_accuracy_data[f'accuracy_foil_{test_prefix}'] = foil_accuracy
                            print(f"    - accuracy_foil_{test_prefix}: {foil_accuracy:.3f} ({len(foil_trials)} trials foil)")
                        else:
                            print(f"    - Aviso: Nenhum trial foil encontrado para {test_prefix}")
                            targetfoil_accuracy_data[f'accuracy_foil_{test_prefix}'] = np.nan
                    else:
                        print(f"    - Aviso: Nenhum valor válido encontrado para targetfoil accuracy de {test_prefix}")
                        targetfoil_accuracy_data[f'accuracy_target_{test_prefix}'] = np.nan
                        targetfoil_accuracy_data[f'accuracy_foil_{test_prefix}'] = np.nan
                else:
                    print(f"    - Aviso: Colunas {targetfoil_col} ou {corr_col} não encontradas")
                    targetfoil_accuracy_data[f'accuracy_target_{test_prefix}'] = np.nan
                    targetfoil_accuracy_data[f'accuracy_foil_{test_prefix}'] = np.nan
            
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
            for key, value in slope_data.items():
                if 'T0' in key:
                    result_dict[key] = value
            for key, value in rt_correct_by_length_data.items():
                if 'T0' in key:
                    result_dict[key] = value
            for key, value in rt_incorrect_by_length_data.items():
                if 'T0' in key:
                    result_dict[key] = value
            for key, value in targetfoil_accuracy_data.items():
                if 'T0' in key:
                    result_dict[key] = value
            for key, value in targetfoil_accuracy_by_length_data.items():
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
            for key, value in slope_data.items():
                if 'T1' in key:
                    result_dict[key] = value
            for key, value in rt_correct_by_length_data.items():
                if 'T1' in key:
                    result_dict[key] = value
            for key, value in rt_incorrect_by_length_data.items():
                if 'T1' in key:
                    result_dict[key] = value
            for key, value in targetfoil_accuracy_data.items():
                if 'T1' in key:
                    result_dict[key] = value
            for key, value in targetfoil_accuracy_by_length_data.items():
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
            for key, value in slope_data.items():
                if 'T2' in key:
                    result_dict[key] = value
            for key, value in rt_correct_by_length_data.items():
                if 'T2' in key:
                    result_dict[key] = value
            for key, value in rt_incorrect_by_length_data.items():
                if 'T2' in key:
                    result_dict[key] = value
            for key, value in targetfoil_accuracy_data.items():
                if 'T2' in key:
                    result_dict[key] = value
            for key, value in targetfoil_accuracy_by_length_data.items():
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