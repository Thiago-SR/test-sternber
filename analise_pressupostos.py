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

def testar_normalidade_eficiencia(df):
    """
    Testa normalidade para eficiência em cada tempo (T0, T1, T2) usando Shapiro-Wilk
    """
    resultados = []
    
    # Colunas de eficiência
    colunas_eficiencia = ['Movimentos_eficiencia_T0', 'Movimentos_eficiencia_T1', 'Movimentos_eficiencia_T2']
    
    for col in colunas_eficiencia:
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
                'Tempo': tempo,
                'n': len(dados_tempo),
                'Shapiro_Stat': stat,
                'Shapiro_p': p_value,
                'Normal': 'Sim' if p_value > 0.05 else 'Não'
            })
        else:
            resultados.append({
                'Tempo': tempo,
                'n': len(dados_tempo),
                'Shapiro_Stat': np.nan,
                'Shapiro_p': np.nan,
                'Normal': 'Dados insuficientes'
            })
    
    return pd.DataFrame(resultados)

def detectar_outliers_eficiencia(df):
    """
    Detecta outliers na eficiência usando IQR e Z-score
    """
    resultados = []
    
    # Colunas de eficiência
    colunas_eficiencia = ['Movimentos_eficiencia_T0', 'Movimentos_eficiencia_T1', 'Movimentos_eficiencia_T2']
    
    for col in colunas_eficiencia:
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

def criar_boxplot_eficiencia(df, output_folder='graficos'):
    """
    Cria boxplot para visualizar outliers na eficiência
    """
    # Criar pasta de gráficos se não existir
    Path(output_folder).mkdir(exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Preparar dados para boxplot
    dados_boxplot = []
    tempos = []
    
    colunas_eficiencia = ['Movimentos_eficiencia_T0', 'Movimentos_eficiencia_T1', 'Movimentos_eficiencia_T2']
    
    for col in colunas_eficiencia:
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
        
        plt.title('Boxplot - Eficiência dos Movimentos', fontsize=16, fontweight='bold')
        plt.xlabel('Tempo de Teste', fontsize=12)
        plt.ylabel('Eficiência dos Movimentos', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Adicionar estatísticas no gráfico
        for i, (tempo, dados) in enumerate(zip(tempos, dados_boxplot)):
            media = dados.mean()
            plt.text(i+1, media, f'Média: {media:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        # Salvar gráfico
        filename = f"{output_folder}/boxplot_eficiencia_movimentos.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Boxplot salvo: {filename}")
    else:
        print(f"   AVISO: Sem dados suficientes para boxplot de eficiência")

def testar_esfericidade_eficiencia(df, id_column):
    """
    Testa esfericidade para eficiência usando Mauchly's test
    """
    try:
        # Preparar dados para pingouin (converter para formato longo)
        anova_data = []
        
        colunas_eficiencia = ['Movimentos_eficiencia_T0', 'Movimentos_eficiencia_T1', 'Movimentos_eficiencia_T2']
        
        for idx, row in df.iterrows():
            participant_id = row[id_column]
            
            for col in colunas_eficiencia:
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
            return {'Erro': 'Sem dados válidos'}
        
        # Verificar se temos dados suficientes para cada tempo
        tempos_unicos = anova_df['time'].unique()
        if len(tempos_unicos) < 3:
            return {'Erro': f'Apenas {len(tempos_unicos)} tempos encontrados. Necessário 3.'}
        
        # Verificar se cada participante tem dados para todos os tempos
        participantes_por_tempo = anova_df.groupby('time')['participant'].nunique()
        if participantes_por_tempo.min() < 3:
            return {'Erro': f'Mínimo de participantes insuficiente: {participantes_por_tempo.min()}'}
        
        # Verificar variabilidade
        if anova_df['value'].std() == 0:
            return {'Erro': 'Sem variabilidade nos dados'}
        
        # Teste de esfericidade
        sphericity = pg.sphericity(anova_df, dv='value', within='time', subject='participant')
        
        # CORREÇÃO: SpherResults é um objeto com atributos diretos, não um DataFrame
        # O objeto SpherResults não tem atributo 'eps', então usamos valores padrão
        return {
            'Mauchly_W': sphericity.W,
            'Mauchly_p': sphericity.pval,
            'Esferico': 'Sim' if sphericity.pval > 0.05 else 'Não',
            'Correcao_GG': 'N/A',  # Não disponível diretamente no SpherResults
            'Correcao_HF': 'N/A'   # Não disponível diretamente no SpherResults
        }
            
    except Exception as e:
        return {
            'Mauchly_W': np.nan,
            'Mauchly_p': np.nan,
            'Esferico': 'Erro',
            'Correcao_GG': np.nan,
            'Correcao_HF': np.nan,
            'Erro': str(e)
        }

def comparacoes_post_hoc_eficiencia(df, id_column):
    """
    Realiza comparações post-hoc para eficiência usando Bonferroni
    """
    try:
        # Preparar dados (converter para formato longo)
        anova_data = []
        
        colunas_eficiencia = ['Movimentos_eficiencia_T0', 'Movimentos_eficiencia_T1', 'Movimentos_eficiencia_T2']
        
        for idx, row in df.iterrows():
            participant_id = row[id_column]
            
            for col in colunas_eficiencia:
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
            return pd.DataFrame([{'Erro': 'Sem dados válidos'}])
        
        print(f"DEBUG Post-hoc: Dados preparados - {len(anova_df)} observações")
        print(f"DEBUG Post-hoc: Tempos únicos: {anova_df['time'].unique()}")
        print(f"DEBUG Post-hoc: Participantes únicos: {anova_df['participant'].nunique()}")
        
        # Verificar se há dados suficientes para comparações
        tempos_unicos = anova_df['time'].unique()
        if len(tempos_unicos) < 2:
            return pd.DataFrame([{'Erro': f'Apenas {len(tempos_unicos)} tempos encontrados. Necessário pelo menos 2.'}])
        
        # Tentar comparações post-hoc com pingouin primeiro
        try:
            posthoc = pg.pairwise_ttests(anova_df, dv='value', within='time', subject='participant', 
                                       padjust='bonf')
            
            print(f"DEBUG Post-hoc: Resultado pairwise_ttests:")
            print(f"DEBUG Post-hoc: Tipo: {type(posthoc)}")
            print(f"DEBUG Post-hoc: Vazio: {posthoc.empty}")
            if not posthoc.empty:
                print(f"DEBUG Post-hoc: Colunas: {posthoc.columns.tolist()}")
                print(f"DEBUG Post-hoc: Shape: {posthoc.shape}")
                print(f"DEBUG Post-hoc: Primeiras linhas:\n{posthoc.head()}")
            
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
                        
                        print(f"DEBUG: {tempo1} vs {tempo2}")
                        print(f"  Diferença de médias: {diff_medias:.3f}")
                        print(f"  IC 95%: [{ic_inferior:.3f}, {ic_superior:.3f}]")
                        print(f"  P-value: {row['p-corr']:.3f}")
                    else:
                        diff_medias = np.nan
                        ic_inferior = np.nan
                        ic_superior = np.nan
                    
                    resultados.append({
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
                print("DEBUG Post-hoc: pingouin falhou, usando método manual")
                raise Exception("pingouin pairwise_ttests falhou")
                
        except Exception as e:
            print(f"DEBUG Post-hoc: pingouin falhou ({str(e)}), usando método manual")
            
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
                            print(f"  Cohen's d (d_z): {cohens_d:.3f}")
                        else:
                            cohens_d = 0
                            print(f"  Cohen's d: não calculável (variância zero ou dados insuficientes)")
                        
                        # Correção de Bonferroni (número de comparações)
                        num_comparacoes = len(tempos) * (len(tempos) - 1) // 2
                        p_corrigido = min(p_value * num_comparacoes, 1.0)
                        
                        print(f"  Diferença de médias: {diff_medias:.3f}")
                        print(f"  IC 95%: [{ic_inferior:.3f}, {ic_superior:.3f}]")
                        print(f"  P-value corrigido: {p_corrigido:.3f}")
                        
                        resultados.append({
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
                return pd.DataFrame([{'Erro': 'Não foi possível realizar comparações manuais'}])
    except Exception as e:
        print(f"DEBUG Post-hoc: Exceção capturada: {str(e)}")
        return pd.DataFrame([{'Erro': str(e)}])

def anova_eficiencia(df, id_column):
    """
    Realiza ANOVA de medidas repetidas para eficiência
    """
    try:
        # Preparar dados (converter para formato longo)
        anova_data = []
        
        colunas_eficiencia = ['Movimentos_eficiencia_T0', 'Movimentos_eficiencia_T1', 'Movimentos_eficiencia_T2']
        
        for idx, row in df.iterrows():
            participant_id = row[id_column]
            
            for col in colunas_eficiencia:
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
            return {'Erro': 'Sem dados válidos'}
        
        # ANOVA de medidas repetidas
        aov = pg.rm_anova(data=anova_df, dv='value', within='time', subject='participant')
        
        # Extrair resultados
        time_row = aov.loc[aov['Source'] == 'time']
        if time_row.empty:
            return {'Erro': 'Não foi possível encontrar resultados para o fator time'}
        
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
            'F': f_value,
            'p_value': p_value,
            'partial_eta_squared': partial_eta_squared,
            'significativo': 'Sim' if p_value < 0.05 else 'Não',
            'tamanho_efeito': 'Grande' if partial_eta_squared >= 0.14 else 'Médio' if partial_eta_squared >= 0.06 else 'Pequeno'
        }
    except Exception as e:
        return {'Erro': str(e)}

def analise_eficiencia_completa(csv_path, output_path=None, criar_graficos=True):
    """
    Realiza análise completa da eficiência dos movimentos
    
    Args:
        csv_path (str): Caminho para o arquivo CSV
        output_path (str): Caminho para salvar resultados Excel
        criar_graficos (bool): Se deve criar boxplots
    """
    
    print("=== ANÁLISE COMPLETA DA EFICIÊNCIA DOS MOVIMENTOS ===\n")
    
    # 1. Leitura dos dados (pular primeira linha que contém descrições)
    print("1. CARREGANDO DADOS...")
    df = pd.read_csv(csv_path, skiprows=1)
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
    
    # Verificar se as colunas de eficiência existem
    colunas_eficiencia = ['Movimentos_eficiencia_T0', 'Movimentos_eficiencia_T1', 'Movimentos_eficiencia_T2']
    colunas_existentes = [col for col in colunas_eficiencia if col in df.columns]
    
    if len(colunas_existentes) == 0:
        print("ERRO: Nenhuma coluna de eficiência encontrada!")
        return
    
    print(f"Colunas de eficiência encontradas: {colunas_existentes}")
    
    # Preparar arquivo de saída
    if output_path is None:
        output_path = 'analise_eficiencia_completa.xlsx'
    
    # Criar pasta para gráficos
    if criar_graficos:
        output_folder = 'graficos_eficiencia'
        Path(output_folder).mkdir(exist_ok=True)
        print(f"\nGráficos serão salvos em: {output_folder}/\n")
    
    # 2. ANÁLISE DA EFICIÊNCIA
    print("2. ANALISANDO EFICIÊNCIA DOS MOVIMENTOS")
    print("-" * 50)
    
    # 2.1 Teste de Normalidade
    print("   Testando normalidade...")
    normalidade = testar_normalidade_eficiencia(df)
    if not normalidade.empty:
        print(f"   Resultados normalidade:")
        for _, row in normalidade.iterrows():
            print(f"     {row['Tempo']}: p = {row['Shapiro_p']:.4f} ({row['Normal']})")
    else:
        print("   AVISO: Não foi possível testar normalidade")
    
    # 2.2 Detecção de Outliers
    print("   Detectando outliers...")
    outliers = detectar_outliers_eficiencia(df)
    if not outliers.empty:
        print(f"   Outliers detectados:")
        for _, row in outliers.iterrows():
            print(f"     {row['Tempo']}: {row['outliers_IQR']} outliers ({row['percent_outliers_IQR']:.1f}%)")
            print(f"       Estatísticas: Média={row['media']:.3f}, DP={row['desvio_padrao']:.3f}")
    else:
        print("   AVISO: Não foi possível detectar outliers")
    
    # 2.3 Criar Boxplot
    if criar_graficos:
        print("   Criando boxplot...")
        criar_boxplot_eficiencia(df, output_folder)
    
    # 2.4 ANOVA de Medidas Repetidas
    print("   Realizando ANOVA de medidas repetidas...")
    anova_result = anova_eficiencia(df, id_column)
    if 'Erro' not in anova_result:
        print(f"   ANOVA: F = {anova_result['F']:.3f}, p = {anova_result['p_value']:.4f}")
        print(f"   Tamanho de efeito (η²) = {anova_result['partial_eta_squared']:.4f} ({anova_result['tamanho_efeito']})")
        print(f"   Resultado: {anova_result['significativo']}")
    else:
        print(f"   ERRO: {anova_result['Erro']}")
    
    # 2.5 Teste de Esfericidade
    print("   Testando esfericidade...")
    esfericidade = testar_esfericidade_eficiencia(df, id_column)
    if 'Erro' not in esfericidade:
        print(f"   Esfericidade: p = {esfericidade['Mauchly_p']:.4f} ({esfericidade['Esferico']})")
        print(f"   Correção GG: {esfericidade['Correcao_GG']}")
        print(f"   Correção HF: {esfericidade['Correcao_HF']}")
    else:
        print(f"   ERRO: {esfericidade['Erro']}")
    
    # 2.6 Comparações Post-hoc
    print("   Realizando comparações post-hoc...")
    posthoc = comparacoes_post_hoc_eficiencia(df, id_column)
    if not posthoc.empty and 'Comparacao' in posthoc.columns:
        print("   Comparações post-hoc:")
        for _, row in posthoc.iterrows():
            print(f"     {row['Comparacao']}:")
            print(f"       Diferença de médias: {row['Diferenca_medias']:.3f}")
            print(f"       P-value corrigido: {row['P_corrigido']:.3f}")
            print(f"       IC 95%: [{row['IC_inferior']:.3f}, {row['IC_superior']:.3f}]")
            print(f"       Significativo: {row['Significativo']}")
            print(f"       Tamanho de efeito (Cohen's d) = {row['Tamanho_efeito']:.3f}")
    else:
        print("   AVISO: Não foi possível realizar comparações post-hoc")
    
    # 3. SALVAR RESULTADOS
    print("\n3. SALVANDO RESULTADOS...")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Resumo geral
        resumo_geral = pd.DataFrame({
            'Analise': ['Normalidade_T0', 'Normalidade_T1', 'Normalidade_T2', 
                       'ANOVA_F', 'ANOVA_p', 'ANOVA_eta2', 'ANOVA_significativo',
                       'Esfericidade_p', 'Esfericidade_resultado'],
            'Valor': [
                normalidade.loc[0, 'Shapiro_p'] if len(normalidade) > 0 else np.nan,
                normalidade.loc[1, 'Shapiro_p'] if len(normalidade) > 1 else np.nan,
                normalidade.loc[2, 'Shapiro_p'] if len(normalidade) > 2 else np.nan,
                anova_result.get('F', np.nan),
                anova_result.get('p_value', np.nan),
                anova_result.get('partial_eta_squared', np.nan),
                anova_result.get('significativo', 'Erro'),
                esfericidade.get('Mauchly_p', np.nan),
                esfericidade.get('Esferico', 'Erro')
            ]
        })
        resumo_geral.to_excel(writer, sheet_name='Resumo_Geral', index=False)
        
        # Detalhes de normalidade
        if not normalidade.empty:
            normalidade.to_excel(writer, sheet_name='Normalidade', index=False)
        
        # Detalhes de outliers
        if not outliers.empty:
            outliers.to_excel(writer, sheet_name='Outliers', index=False)
        
        # Comparações post-hoc
        if not posthoc.empty and 'Comparacao' in posthoc.columns:
            posthoc.to_excel(writer, sheet_name='PostHoc', index=False)
    
    print(f"\nAnálise completa salva em: {output_path}")
    if criar_graficos:
        print(f"Gráfico salvo em: {output_folder}/")
    
    return output_path

def main():
    """
    Função principal
    """
    csv_path = '03_analises_combinadas/todos_usuarios_analises.csv'
    
    if not Path(csv_path).exists():
        print(f"ERRO: Arquivo não encontrado: {csv_path}")
        print("Por favor, execute primeiro o pipeline principal.")
        return
    
    print("Este script analisa especificamente a EFICIÊNCIA DOS MOVIMENTOS.")
    print("Inclui:")
    print("- Teste de normalidade (Shapiro-Wilk)")
    print("- Detecção de outliers (IQR e Z-score)")
    print("- Boxplot para visualização")
    print("- ANOVA de medidas repetidas")
    print("- Teste de esfericidade (Mauchly)")
    print("- Comparações post-hoc (Bonferroni)")
    print()
    
    analise_eficiencia_completa(csv_path, criar_graficos=True)

if __name__ == "__main__":
    main()