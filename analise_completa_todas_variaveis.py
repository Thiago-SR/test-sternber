import pandas as pd
import pingouin as pg
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest
import warnings
warnings.filterwarnings('ignore')

def identificar_variaveis_unicas(df):
    """
    Identifica todas as variáveis únicas que têm dados para T0, T1 e T2
    """
    # Obter todas as colunas exceto 'id'
    colunas = [col for col in df.columns if col != 'id']
    
    # Identificar variáveis únicas (removendo sufixos T0, T1, T2)
    variaveis_unicas = set()
    
    for col in colunas:
        if col.endswith('_T0'):
            variavel_base = col[:-3]  # Remove '_T0'
            # Verificar se existe T1 e T2 para esta variável
            col_t1 = variavel_base + '_T1'
            col_t2 = variavel_base + '_T2'
            if col_t1 in df.columns and col_t2 in df.columns:
                variaveis_unicas.add(variavel_base)
        elif col.endswith('_T1'):
            variavel_base = col[:-3]  # Remove '_T1'
            # Verificar se existe T0 e T2 para esta variável
            col_t0 = variavel_base + '_T0'
            col_t2 = variavel_base + '_T2'
            if col_t0 in df.columns and col_t2 in df.columns:
                variaveis_unicas.add(variavel_base)
        elif col.endswith('_T2'):
            variavel_base = col[:-3]  # Remove '_T2'
            # Verificar se existe T0 e T1 para esta variável
            col_t0 = variavel_base + '_T0'
            col_t1 = variavel_base + '_T1'
            if col_t0 in df.columns and col_t1 in df.columns:
                variaveis_unicas.add(variavel_base)
    
    return sorted(list(variaveis_unicas))

def testar_normalidade_variavel(df, variavel_base):
    """
    Testa normalidade para uma variável específica em cada tempo (T0, T1, T2)
    """
    resultados = []
    
    # Colunas da variável
    colunas_variavel = [f'{variavel_base}_T0', f'{variavel_base}_T1', f'{variavel_base}_T2']
    
    for col in colunas_variavel:
        if col not in df.columns:
            continue
            
        # Extrair tempo da coluna
        if col.endswith('_T0'):
            tempo = 'T0'
        elif col.endswith('_T1'):
            tempo = 'T1'
        elif col.endswith('_T2'):
            tempo = 'T2'
        else:
            continue
            
        dados_tempo = df[col].dropna()
        
        # Converter para numérico, ignorando strings
        dados_tempo = pd.to_numeric(dados_tempo, errors='coerce').dropna()
        
        if len(dados_tempo) >= 3:  # Mínimo para Shapiro-Wilk
            stat, p_value = shapiro(dados_tempo)
            resultados.append({
                'Variavel': variavel_base,
                'Tempo': tempo,
                'n': len(dados_tempo),
                'Shapiro_Stat': stat,
                'Shapiro_p': p_value,
                'Normal': 'Sim' if p_value > 0.05 else 'Não'
            })
        else:
            resultados.append({
                'Variavel': variavel_base,
                'Tempo': tempo,
                'n': len(dados_tempo),
                'Shapiro_Stat': np.nan,
                'Shapiro_p': np.nan,
                'Normal': 'Dados insuficientes'
            })
    
    return pd.DataFrame(resultados)

def detectar_outliers_variavel(df, variavel_base):
    """
    Detecta outliers para uma variável específica usando IQR e Z-score
    """
    resultados = []
    
    # Colunas da variável
    colunas_variavel = [f'{variavel_base}_T0', f'{variavel_base}_T1', f'{variavel_base}_T2']
    
    for col in colunas_variavel:
        if col not in df.columns:
            continue
            
        # Extrair tempo da coluna
        if col.endswith('_T0'):
            tempo = 'T0'
        elif col.endswith('_T1'):
            tempo = 'T1'
        elif col.endswith('_T2'):
            tempo = 'T2'
        else:
            continue
            
        dados_tempo = df[col].dropna()
        
        # Converter para numérico, ignorando strings
        dados_tempo = pd.to_numeric(dados_tempo, errors='coerce').dropna()
        
        if len(dados_tempo) > 0:
            # Método IQR
            Q1 = dados_tempo.quantile(0.25)
            Q3 = dados_tempo.quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            
            outliers_iqr = dados_tempo[(dados_tempo < limite_inferior) | (dados_tempo > limite_superior)]
            
            # Método Z-score
            if len(dados_tempo) > 1:  # Precisa de pelo menos 2 valores para calcular z-score
                z_scores = np.abs(stats.zscore(dados_tempo))
                outliers_z = dados_tempo[z_scores > 3]
            else:
                outliers_z = pd.Series(dtype=float)
            
            resultados.append({
                'Variavel': variavel_base,
                'Tempo': tempo,
                'n_total': len(dados_tempo),
                'outliers_IQR': len(outliers_iqr),
                'outliers_Zscore': len(outliers_z),
                'percent_outliers_IQR': (len(outliers_iqr) / len(dados_tempo)) * 100,
                'percent_outliers_Zscore': (len(outliers_z) / len(dados_tempo)) * 100,
                'media': dados_tempo.mean(),
                'desvio_padrao': dados_tempo.std(),
                'min': dados_tempo.min(),
                'max': dados_tempo.max()
            })
    
    return pd.DataFrame(resultados)

def criar_boxplot_variavel(df, variavel_base, output_folder='graficos'):
    """
    Cria boxplot para visualizar outliers de uma variável específica
    """
    # Criar pasta de gráficos se não existir
    Path(output_folder).mkdir(exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Preparar dados para boxplot
    dados_boxplot = []
    tempos = []
    
    colunas_variavel = [f'{variavel_base}_T0', f'{variavel_base}_T1', f'{variavel_base}_T2']
    
    for col in colunas_variavel:
        if col not in df.columns:
            continue
            
        # Extrair tempo da coluna
        if col.endswith('_T0'):
            tempo = 'T0'
        elif col.endswith('_T1'):
            tempo = 'T1'
        elif col.endswith('_T2'):
            tempo = 'T2'
        else:
            continue
            
        dados_tempo = df[col].dropna()
        
        # Converter para numérico, ignorando strings
        dados_tempo = pd.to_numeric(dados_tempo, errors='coerce').dropna()
        
        if len(dados_tempo) > 0:
            dados_boxplot.append(dados_tempo)
            tempos.append(tempo)
    
    if dados_boxplot:
        # Criar boxplot
        bp = plt.boxplot(dados_boxplot, labels=tempos, patch_artist=True)
        
        # Colorir os boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.title(f'Boxplot - {variavel_base}', fontsize=16, fontweight='bold')
        plt.xlabel('Tempo de Teste', fontsize=12)
        plt.ylabel(variavel_base, fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Adicionar estatísticas no gráfico
        for i, (tempo, dados) in enumerate(zip(tempos, dados_boxplot)):
            media = dados.mean()
            plt.text(i+1, media, f'Média: {media:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        # Salvar gráfico
        filename = f"{output_folder}/boxplot_{variavel_base}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    else:
        plt.close()
        return None

def testar_esfericidade_variavel(df, variavel_base, id_column):
    """
    Testa esfericidade para uma variável específica usando Mauchly's test
    """
    try:
        # Preparar dados para pingouin (converter para formato longo)
        anova_data = []
        
        colunas_variavel = [f'{variavel_base}_T0', f'{variavel_base}_T1', f'{variavel_base}_T2']
        
        for idx, row in df.iterrows():
            participant_id = row[id_column]
            
            for col in colunas_variavel:
                if col not in df.columns:
                    continue
                    
                # Extrair tempo da coluna
                if col.endswith('_T0'):
                    tempo = 'T0'
                elif col.endswith('_T1'):
                    tempo = 'T1'
                elif col.endswith('_T2'):
                    tempo = 'T2'
                else:
                    continue
                    
                value = row[col]
                
                # Converter para numérico
                try:
                    value = pd.to_numeric(value, errors='coerce')
                    if pd.notna(value):
                        anova_data.append({
                            'participant': participant_id,
                            'time': tempo,
                            'value': value
                        })
                except:
                    continue
        
        anova_df = pd.DataFrame(anova_data)
        
        if len(anova_df) == 0:
            return {'Variavel': variavel_base, 'Erro': 'Sem dados válidos'}
        
        # Verificar se temos dados suficientes para cada tempo
        tempos_unicos = anova_df['time'].unique()
        if len(tempos_unicos) < 3:
            return {'Variavel': variavel_base, 'Erro': f'Apenas {len(tempos_unicos)} tempos encontrados. Necessário 3.'}
        
        # Verificar se cada participante tem dados para todos os tempos
        participantes_por_tempo = anova_df.groupby('time')['participant'].nunique()
        if participantes_por_tempo.min() < 3:
            return {'Variavel': variavel_base, 'Erro': f'Mínimo de participantes insuficiente: {participantes_por_tempo.min()}'}
        
        # Verificar variabilidade
        if anova_df['value'].std() == 0:
            return {'Variavel': variavel_base, 'Erro': 'Sem variabilidade nos dados'}
        
        # Teste de esfericidade
        sphericity = pg.sphericity(anova_df, dv='value', within='time', subject='participant')
        
        return {
            'Variavel': variavel_base,
            'Mauchly_W': sphericity.W,
            'Mauchly_p': sphericity.pval,
            'Esferico': 'Sim' if sphericity.pval > 0.05 else 'Não',
            'Correcao_GG': 'N/A',
            'Correcao_HF': 'N/A'
        }
            
    except Exception as e:
        return {
            'Variavel': variavel_base,
            'Mauchly_W': np.nan,
            'Mauchly_p': np.nan,
            'Esferico': 'Erro',
            'Correcao_GG': np.nan,
            'Correcao_HF': np.nan,
            'Erro': str(e)
        }

def comparacoes_post_hoc_variavel(df, variavel_base, id_column):
    """
    Realiza comparações post-hoc para uma variável específica usando Bonferroni
    """
    try:
        # Preparar dados (converter para formato longo)
        anova_data = []
        
        colunas_variavel = [f'{variavel_base}_T0', f'{variavel_base}_T1', f'{variavel_base}_T2']
        
        for idx, row in df.iterrows():
            participant_id = row[id_column]
            
            for col in colunas_variavel:
                if col not in df.columns:
                    continue
                    
                # Extrair tempo da coluna
                if col.endswith('_T0'):
                    tempo = 'T0'
                elif col.endswith('_T1'):
                    tempo = 'T1'
                elif col.endswith('_T2'):
                    tempo = 'T2'
                else:
                    continue
                    
                value = row[col]
                
                # Converter para numérico
                try:
                    value = pd.to_numeric(value, errors='coerce')
                    if pd.notna(value):
                        anova_data.append({
                            'participant': participant_id,
                            'time': tempo,
                            'value': value
                        })
                except:
                    continue
        
        anova_df = pd.DataFrame(anova_data)
        
        if len(anova_df) == 0:
            return pd.DataFrame([{'Variavel': variavel_base, 'Erro': 'Sem dados válidos'}])
        
        # Verificar se há dados suficientes para comparações
        tempos_unicos = anova_df['time'].unique()
        if len(tempos_unicos) < 2:
            return pd.DataFrame([{'Variavel': variavel_base, 'Erro': f'Apenas {len(tempos_unicos)} tempos encontrados. Necessário pelo menos 2.'}])
        
        # Tentar comparações post-hoc com pingouin primeiro
        try:
            posthoc = pg.pairwise_ttests(anova_df, dv='value', within='time', subject='participant', 
                                       padjust='bonf')
            
            # Verificar se o resultado tem as colunas esperadas
            if not posthoc.empty and all(col in posthoc.columns for col in ['A', 'B', 'T', 'p-unc', 'p-corr']):
                # Organizar resultados do pingouin
                resultados = []
                for _, row in posthoc.iterrows():
                    # Usar 'hedges' se disponível, senão calcular manualmente
                    tamanho_efeito = row.get('hedges', np.nan)
                    if pd.isna(tamanho_efeito):
                        # Calcular Cohen's d aproximado
                        tamanho_efeito = row['T'] / np.sqrt(row['dof'] + 1)
                    
                    # Calcular diferença de médias e IC para cada comparação
                    tempo1, tempo2 = row['A'], row['B']
                    
                    # Encontrar participantes comuns para cálculo correto
                    participantes_tempo1 = anova_df[anova_df['time'] == tempo1]['participant'].values
                    participantes_tempo2 = anova_df[anova_df['time'] == tempo2]['participant'].values
                    participantes_comuns = np.intersect1d(participantes_tempo1, participantes_tempo2)
                    
                    if len(participantes_comuns) >= 3:
                        # Reorganizar dados para participantes comuns
                        dados1_comuns = []
                        dados2_comuns = []
                        for p in participantes_comuns:
                            val1 = anova_df[(anova_df['time'] == tempo1) & (anova_df['participant'] == p)]['value'].iloc[0]
                            val2 = anova_df[(anova_df['time'] == tempo2) & (anova_df['participant'] == p)]['value'].iloc[0]
                            dados1_comuns.append(val1)
                            dados2_comuns.append(val2)
                        
                        dados1_comuns = np.array(dados1_comuns)
                        dados2_comuns = np.array(dados2_comuns)
                        
                        # Diferença de médias (T1 - T2)
                        diff_medias = np.mean(dados1_comuns) - np.mean(dados2_comuns)
                        
                        # Intervalo de confiança para diferença de médias
                        diff_individual = dados1_comuns - dados2_comuns
                        n = len(diff_individual)
                        se_diff = np.std(diff_individual, ddof=1) / np.sqrt(n)
                        
                        # IC 95% usando distribuição t
                        from scipy.stats import t
                        t_critico = t.ppf(0.975, df=n-1)  # 95% CI
                        ic_inferior = diff_medias - t_critico * se_diff
                        ic_superior = diff_medias + t_critico * se_diff
                    else:
                        diff_medias = np.nan
                        ic_inferior = np.nan
                        ic_superior = np.nan
                    
                    resultados.append({
                        'Variavel': variavel_base,
                        'Comparacao': f"{row['A']} vs {row['B']}",
                        'Diferenca_medias': diff_medias,
                        'P_corrigido': row['p-corr'],
                        'IC_inferior': ic_inferior,
                        'IC_superior': ic_superior,
                        'T_statistic': row['T'],
                        'p_value': row['p-unc'],
                        'Significativo': 'Sim' if row['p-corr'] < 0.05 else 'Não',
                        'Tamanho_efeito': tamanho_efeito
                    })
                return pd.DataFrame(resultados)
            else:
                raise Exception("pingouin pairwise_ttests falhou")
                
        except Exception as e:
            # Método manual usando scipy.stats
            from scipy.stats import ttest_rel
            
            resultados = []
            tempos = sorted(anova_df['time'].unique())
            
            # Fazer todas as comparações pareadas
            for i in range(len(tempos)):
                for j in range(i+1, len(tempos)):
                    tempo1, tempo2 = tempos[i], tempos[j]
                    
                    # Obter dados para cada tempo
                    dados_tempo1 = anova_df[anova_df['time'] == tempo1]['value'].values
                    dados_tempo2 = anova_df[anova_df['time'] == tempo2]['value'].values
                    
                    # Garantir que temos os mesmos participantes
                    participantes_tempo1 = anova_df[anova_df['time'] == tempo1]['participant'].values
                    participantes_tempo2 = anova_df[anova_df['time'] == tempo2]['participant'].values
                    
                    # Encontrar participantes comuns
                    participantes_comuns = np.intersect1d(participantes_tempo1, participantes_tempo2)
                    
                    if len(participantes_comuns) >= 3:  # Mínimo para t-test
                        # Reorganizar dados para participantes comuns
                        dados1_comuns = []
                        dados2_comuns = []
                        
                        for p in participantes_comuns:
                            val1 = anova_df[(anova_df['time'] == tempo1) & (anova_df['participant'] == p)]['value'].iloc[0]
                            val2 = anova_df[(anova_df['time'] == tempo2) & (anova_df['participant'] == p)]['value'].iloc[0]
                            dados1_comuns.append(val1)
                            dados2_comuns.append(val2)
                        
                        dados1_comuns = np.array(dados1_comuns)
                        dados2_comuns = np.array(dados2_comuns)
                        
                        # T-test pareado
                        t_stat, p_value = ttest_rel(dados1_comuns, dados2_comuns)
                        
                        # Diferença de médias
                        diff_medias = np.mean(dados1_comuns) - np.mean(dados2_comuns)
                        
                        # Intervalo de confiança para diferença de médias
                        diff_individual = dados1_comuns - dados2_comuns
                        n = len(diff_individual)
                        se_diff = np.std(diff_individual, ddof=1) / np.sqrt(n)
                        
                        # IC 95% usando distribuição t
                        from scipy.stats import t
                        t_critico = t.ppf(0.975, df=n-1)  # 95% CI
                        ic_inferior = diff_medias - t_critico * se_diff
                        ic_superior = diff_medias + t_critico * se_diff
                        
                        # Cohen's d para dados pareados (d_z) - mais preciso
                        diff = dados1_comuns - dados2_comuns
                        
                        # Verificar se há missing values nas diferenças
                        diff_validos = diff[~np.isnan(diff)]
                        
                        if len(diff_validos) > 1 and np.std(diff_validos, ddof=1) > 0:
                            cohens_d = np.mean(diff_validos) / np.std(diff_validos, ddof=1)
                        else:
                            cohens_d = 0
                        
                        # Correção de Bonferroni (número de comparações)
                        num_comparacoes = len(tempos) * (len(tempos) - 1) // 2
                        p_corrigido = min(p_value * num_comparacoes, 1.0)
                        
                        resultados.append({
                            'Variavel': variavel_base,
                            'Comparacao': f"{tempo1} vs {tempo2}",
                            'Diferenca_medias': diff_medias,
                            'P_corrigido': p_corrigido,
                            'IC_inferior': ic_inferior,
                            'IC_superior': ic_superior,
                            'T_statistic': t_stat,
                            'p_value': p_value,
                            'Significativo': 'Sim' if p_corrigido < 0.05 else 'Não',
                            'Tamanho_efeito': cohens_d
                        })
            
            if resultados:
                return pd.DataFrame(resultados)
            else:
                return pd.DataFrame([{'Variavel': variavel_base, 'Erro': 'Não foi possível realizar comparações manuais'}])
    except Exception as e:
        return pd.DataFrame([{'Variavel': variavel_base, 'Erro': str(e)}])

def anova_variavel(df, variavel_base, id_column):
    """
    Realiza ANOVA de medidas repetidas para uma variável específica
    """
    try:
        # Preparar dados (converter para formato longo)
        anova_data = []
        
        colunas_variavel = [f'{variavel_base}_T0', f'{variavel_base}_T1', f'{variavel_base}_T2']
        
        for idx, row in df.iterrows():
            participant_id = row[id_column]
            
            for col in colunas_variavel:
                if col not in df.columns:
                    continue
                    
                # Extrair tempo da coluna
                if col.endswith('_T0'):
                    tempo = 'T0'
                elif col.endswith('_T1'):
                    tempo = 'T1'
                elif col.endswith('_T2'):
                    tempo = 'T2'
                else:
                    continue
                    
                value = row[col]
                
                # Converter para numérico
                try:
                    value = pd.to_numeric(value, errors='coerce')
                    if pd.notna(value):
                        anova_data.append({
                            'participant': participant_id,
                            'time': tempo,
                            'value': value
                        })
                except:
                    continue
        
        anova_df = pd.DataFrame(anova_data)
        
        if len(anova_df) == 0:
            return {'Variavel': variavel_base, 'Erro': 'Sem dados válidos'}
        
        # ANOVA de medidas repetidas
        aov = pg.rm_anova(data=anova_df, dv='value', within='time', subject='participant')
        
        # Extrair resultados
        time_row = aov.loc[aov['Source'] == 'time']
        if time_row.empty:
            return {'Variavel': variavel_base, 'Erro': 'Não foi possível encontrar resultados para o fator time'}
        
        p_value = time_row['p-unc'].iloc[0]
        f_value = time_row['F'].iloc[0]
        
        # Calcular partial eta squared
        if 'ng2' in time_row.columns:
            partial_eta_squared = time_row['ng2'].iloc[0]
        else:
            df1 = time_row['ddof1'].iloc[0]
            df2 = time_row['ddof2'].iloc[0]
            partial_eta_squared = (f_value * df1) / (f_value * df1 + df2)
        
        return {
            'Variavel': variavel_base,
            'F': f_value,
            'p_value': p_value,
            'partial_eta_squared': partial_eta_squared,
            'significativo': 'Sim' if p_value < 0.05 else 'Não',
            'tamanho_efeito': 'Grande' if partial_eta_squared >= 0.14 else 'Médio' if partial_eta_squared >= 0.06 else 'Pequeno'
        }
    except Exception as e:
        return {'Variavel': variavel_base, 'Erro': str(e)}

def analise_completa_todas_variaveis(csv_path, output_path=None, criar_graficos=True):
    """
    Realiza análise completa para todas as variáveis dos 3 momentos
    
    Args:
        csv_path (str): Caminho para o arquivo CSV
        output_path (str): Caminho para salvar resultados Excel
        criar_graficos (bool): Se deve criar boxplots
    """
    
    print("=== ANÁLISE COMPLETA DE TODAS AS VARIÁVEIS ===\n")
    
    # 1. Leitura dos dados
    print("1. CARREGANDO DADOS...")
    df = pd.read_csv(csv_path)
    print(f"Dados carregados: {df.shape[0]} participantes, {df.shape[1]} colunas")
    
    # Identificar coluna de ID
    id_column = 'id'
    if id_column not in df.columns:
        possible_id_cols = [col for col in df.columns if 'participante' in col.lower() or 'id' in col.lower()]
        if possible_id_cols:
            id_column = possible_id_cols[0]
        else:
            raise ValueError("Coluna de ID não encontrada")
    
    print(f"Coluna de ID identificada: {id_column}")
    
    # 2. Identificar variáveis únicas
    print("\n2. IDENTIFICANDO VARIÁVEIS...")
    variaveis_unicas = identificar_variaveis_unicas(df)
    print(f"Encontradas {len(variaveis_unicas)} variáveis únicas com dados para T0, T1 e T2:")
    for i, var in enumerate(variaveis_unicas, 1):
        print(f"  {i:2d}. {var}")
    
    # Preparar arquivo de saída
    if output_path is None:
        output_path = 'analise_completa_todas_variaveis.xlsx'
    
    # Criar pasta para gráficos
    if criar_graficos:
        output_folder = 'graficos_todas_variaveis'
        Path(output_folder).mkdir(exist_ok=True)
        print(f"\nGr\u00e1ficos ser\u00e3o salvos em: {output_folder}/\n")
    
    # 3. ANÁLISE DE CADA VARIÁVEL
    print("3. ANALISANDO CADA VARIÁVEL")
    print("=" * 60)
    
    # Listas para armazenar resultados de todas as variáveis
    todos_normalidade = []
    todos_outliers = []
    todos_esfericidade = []
    todos_posthoc = []
    todos_anova = []
    
    for i, variavel_base in enumerate(variaveis_unicas, 1):
        print(f"\n{i:2d}/{len(variaveis_unicas)} - Analisando: {variavel_base}")
        print("-" * 50)
        
        # 3.1 Teste de Normalidade
        print("   Testando normalidade...")
        normalidade = testar_normalidade_variavel(df, variavel_base)
        if not normalidade.empty:
            todos_normalidade.append(normalidade)
            for _, row in normalidade.iterrows():
                print(f"     {row['Tempo']}: p = {row['Shapiro_p']:.4f} ({row['Normal']})")
        else:
            print("     AVISO: Não foi possível testar normalidade")
        
        # 3.2 Detecção de Outliers
        print("   Detectando outliers...")
        outliers = detectar_outliers_variavel(df, variavel_base)
        if not outliers.empty:
            todos_outliers.append(outliers)
            for _, row in outliers.iterrows():
                print(f"     {row['Tempo']}: {row['outliers_IQR']} outliers ({row['percent_outliers_IQR']:.1f}%)")
        else:
            print("     AVISO: Não foi possível detectar outliers")
        
        # 3.3 Criar Boxplot
        if criar_graficos:
            print("   Criando boxplot...")
            filename = criar_boxplot_variavel(df, variavel_base, output_folder)
            if filename:
                print(f"     Boxplot salvo: {filename}")
            else:
                print("     AVISO: Não foi possível criar boxplot")
        
        # 3.4 ANOVA de Medidas Repetidas
        print("   Realizando ANOVA de medidas repetidas...")
        anova_result = anova_variavel(df, variavel_base, id_column)
        if 'Erro' not in anova_result:
            todos_anova.append(pd.DataFrame([anova_result]))
            print(f"     ANOVA: F = {anova_result['F']:.3f}, p = {anova_result['p_value']:.4f}")
            print(f"     Tamanho de efeito (η²) = {anova_result['partial_eta_squared']:.4f} ({anova_result['tamanho_efeito']})")
            print(f"     Resultado: {anova_result['significativo']}")
        else:
            print(f"     ERRO: {anova_result['Erro']}")
        
        # 3.5 Teste de Esfericidade
        print("   Testando esfericidade...")
        esfericidade = testar_esfericidade_variavel(df, variavel_base, id_column)
        if 'Erro' not in esfericidade:
            todos_esfericidade.append(pd.DataFrame([esfericidade]))
            print(f"     Esfericidade: p = {esfericidade['Mauchly_p']:.4f} ({esfericidade['Esferico']})")
        else:
            print(f"     ERRO: {esfericidade['Erro']}")
        
        # 3.6 Comparações Post-hoc
        print("   Realizando comparações post-hoc...")
        posthoc = comparacoes_post_hoc_variavel(df, variavel_base, id_column)
        if not posthoc.empty and 'Comparacao' in posthoc.columns:
            todos_posthoc.append(posthoc)
            print("     Comparações post-hoc:")
            for _, row in posthoc.iterrows():
                print(f"       {row['Comparacao']}: p = {row['P_corrigido']:.3f} ({row['Significativo']})")
        else:
            print("     AVISO: Não foi possível realizar comparações post-hoc")
    
    # 4. SALVAR RESULTADOS
    print("\n4. SALVANDO RESULTADOS...")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Resumo geral
        resumo_geral = pd.DataFrame({
            'Variavel': variaveis_unicas,
            'Total_Variaveis': len(variaveis_unicas)
        })
        resumo_geral.to_excel(writer, sheet_name='Resumo_Geral', index=False)
        
        # Detalhes de normalidade
        if todos_normalidade:
            normalidade_completa = pd.concat(todos_normalidade, ignore_index=True)
            normalidade_completa.to_excel(writer, sheet_name='Normalidade', index=False)
        
        # Detalhes de outliers
        if todos_outliers:
            outliers_completo = pd.concat(todos_outliers, ignore_index=True)
            outliers_completo.to_excel(writer, sheet_name='Outliers', index=False)
        
        # Detalhes de ANOVA
        if todos_anova:
            anova_completa = pd.concat(todos_anova, ignore_index=True)
            anova_completa.to_excel(writer, sheet_name='ANOVA', index=False)
        
        # Detalhes de esfericidade
        if todos_esfericidade:
            esfericidade_completa = pd.concat(todos_esfericidade, ignore_index=True)
            esfericidade_completa.to_excel(writer, sheet_name='Esfericidade', index=False)
        
        # Comparações post-hoc
        if todos_posthoc:
            posthoc_completo = pd.concat(todos_posthoc, ignore_index=True)
            posthoc_completo.to_excel(writer, sheet_name='PostHoc', index=False)
    
    print(f"\nAnálise completa salva em: {output_path}")
    if criar_graficos:
        print(f"Gráficos salvos em: {output_folder}/")
    
    return output_path

def main():
    """
    Função principal
    """
    csv_path = 'analises.csv'
    
    if not Path(csv_path).exists():
        print(f"ERRO: Arquivo não encontrado: {csv_path}")
        return
    
    print("Este script analisa TODAS AS VARIÁVEIS dos dados Sternberg.")
    print("Inclui para cada variável:")
    print("- Teste de normalidade (Shapiro-Wilk)")
    print("- Detecção de outliers (IQR e Z-score)")
    print("- Boxplot para visualização")
    print("- ANOVA de medidas repetidas")
    print("- Teste de esfericidade (Mauchly)")
    print("- Comparações post-hoc (Bonferroni)")
    print()
    
    analise_completa_todas_variaveis(csv_path, criar_graficos=True)

if __name__ == "__main__":
    main()
