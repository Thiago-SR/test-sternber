import pandas as pd
import pingouin as pg
import numpy as np
from pathlib import Path
import re

def realizar_anova_medidas_repetidas(csv_path, output_path=None):
    """
    Realiza ANOVA de medidas repetidas para todas as variáveis em um arquivo CSV.
    
    Args:
        csv_path (str): Caminho para o arquivo CSV com os dados
        output_path (str): Caminho para salvar o arquivo Excel com resultados (opcional)
    
    Returns:
        pd.DataFrame: DataFrame com os resultados das ANOVAs
    """
    
    # 1. Leitura do arquivo CSV
    print("Lendo arquivo CSV...")
    # Ler o arquivo tentando detectar automaticamente cabeçalho/linha descritiva
    df = None
    attempts = [0, 1]
    last_error = None
    for skip in attempts:
        try:
            temp_df = pd.read_csv(csv_path, skiprows=skip)
            if 'id' in [c.lower() for c in temp_df.columns]:
                # Padronizar nome da coluna de id para 'id'
                rename_map = {c: 'id' for c in temp_df.columns if c.lower() == 'id'}
                df = temp_df.rename(columns=rename_map)
                break
        except Exception as e:
            last_error = e
            continue
    if df is None:
        raise ValueError(f"Falha ao ler CSV. Último erro: {last_error}")
    print(f"Dados carregados: {df.shape[0]} participantes, {df.shape[1]} colunas")
    
    # Identificar a coluna de ID
    id_column = 'id'  # Assumindo que a coluna se chama 'id'
    if id_column not in df.columns:
        # Tentar encontrar coluna de ID
        possible_id_cols = [col for col in df.columns if 'id' in col.lower() or 'participante' in col.lower()]
        if possible_id_cols:
            id_column = possible_id_cols[0]
            print(f"Coluna de ID identificada: {id_column}")
        else:
            raise ValueError("Não foi possível encontrar a coluna de ID")
    
    # 2. Identificar variáveis e momentos de teste
    print("\nIdentificando variáveis...")
    # Detectar padrões _T0/_T1/_T2 em qualquer posição do nome da coluna
    # e agrupar removendo o token _T[0-2]
    variable_groups = {}  # base_variable -> { 'T0': col, 'T1': col, 'T2': col }
    time_pattern = re.compile(r'_T([012])(?=(_|$))')
    
    for col in df.columns:
        if col == id_column:
            continue
        match = time_pattern.search(col)
        if not match:
            continue
        time_key = f"T{match.group(1)}"
        # Remover apenas o token _T[0-2], mantendo quaisquer sufixos (ex.: _2)
        base_name = time_pattern.sub('', col)
        # Normalizações leves
        base_name = re.sub(r'__+', '_', base_name).strip('_')
        if base_name not in variable_groups:
            variable_groups[base_name] = {}
        variable_groups[base_name][time_key] = col
    
    total_groups = len(variable_groups)
    complete_groups = sum(1 for v in variable_groups.values() if set(v.keys()) == {'T0', 'T1', 'T2'})
    print(f"Encontradas {total_groups} variáveis candidatas; {complete_groups} com T0, T1 e T2 presentes.")
    
    # 3. Realizar ANOVA para cada variável
    print("\nRealizando ANOVAs de medidas repetidas...")
    resultados = []
    
    for variable_name, time_to_col in variable_groups.items():
        print(f"\nAnalisando: {variable_name}")
        # Verificar se temos as 3 colunas necessárias (T0, T1, T2)
        if set(time_to_col.keys()) != {'T0', 'T1', 'T2'}:
            print(f"  AVISO: Variável {variable_name} não tem exatamente 3 momentos (T0, T1, T2). Pulando...")
            continue
        # Ordenar colunas por tempo
        columns = [time_to_col['T0'], time_to_col['T1'], time_to_col['T2']]
        print(f"  Colunas: {columns}")
        
        # Preparar dados para ANOVA
        # Criar DataFrame no formato longo (long format) necessário para pingouin
        anova_data = []
        
        for idx, row in df.iterrows():
            participant_id = row[id_column]
            # Adicionar dados para cada momento na ordem T0, T1, T2
            for t_idx, t in enumerate(['T0', 'T1', 'T2']):
                col = time_to_col[t]
                value = row[col]
                if pd.notna(value):
                    anova_data.append({'participant': participant_id, 'time': t, 'value': value})
        
        anova_df = pd.DataFrame(anova_data)
        
        if len(anova_df) == 0:
            print(f"  AVISO: Nenhum dado válido para {variable_name}. Pulando...")
            continue
        
        # Mostrar informações sobre os dados
        print(f"  Dados preparados: {len(anova_df)} observações")
        print(f"  Valores únicos por momento: {anova_df['time'].value_counts().to_dict()}")
        
        # Verificar se há variabilidade nos dados
        if anova_df['value'].std() == 0:
            print(f"  AVISO: Sem variabilidade nos dados para {variable_name}. Pulando...")
            continue
        
        try:
            # Realizar ANOVA de medidas repetidas
            aov = pg.rm_anova(data=anova_df, dv='value', within='time', subject='participant')
            
            # Verificar se a ANOVA foi bem-sucedida
            if aov.empty or 'time' not in aov['Source'].values:
                raise ValueError("ANOVA não retornou resultados válidos")
            
            # Extrair resultados
            time_row = aov.loc[aov['Source'] == 'time']
            if time_row.empty:
                raise ValueError("Não foi possível encontrar resultados para o fator 'time'")
            
            p_value = time_row['p-unc'].iloc[0]
            f_value = time_row['F'].iloc[0]
            
            # Calcular partial eta squared (tamanho de efeito)
            # Usar a coluna 'ng2' que é o partial eta squared calculado pelo pingouin
            if 'ng2' in time_row.columns:
                partial_eta_squared = time_row['ng2'].iloc[0]
            else:
                # Fallback: calcular usando F e graus de liberdade
                df1 = time_row['ddof1'].iloc[0]
                df2 = time_row['ddof2'].iloc[0]
                partial_eta_squared = (f_value * df1) / (f_value * df1 + df2)
            
            # Adicionar resultado
            resultados.append({
                'Variavel': variable_name,
                'F': f_value,
                'p_value': p_value,
                'partial_eta_squared': partial_eta_squared,
                'significancia': 'Sim' if p_value < 0.05 else 'Não',
                'tamanho_efeito': 'Grande' if partial_eta_squared >= 0.14 else 'Médio' if partial_eta_squared >= 0.06 else 'Pequeno'
            })
            
            print(f"  OK: ANOVA concluída - p = {p_value:.4f}, eta2 = {partial_eta_squared:.4f}")
            
        except Exception as e:
            print(f"  ERRO: Erro na ANOVA para {variable_name}: {str(e)}")
            # Adicionar resultado com erro
            resultados.append({
                'Variavel': variable_name,
                'F': np.nan,
                'p_value': np.nan,
                'partial_eta_squared': np.nan,
                'significancia': 'Erro',
                'tamanho_efeito': 'Erro'
            })
    
    # 4️⃣ Criar DataFrame com resultados
    resultados_df = pd.DataFrame(resultados)
    
    # Ordenar por p-value (menor primeiro)
    resultados_df = resultados_df.sort_values('p_value', na_position='last')
    
    print(f"\nResumo dos resultados:")
    print(f"Total de variáveis analisadas: {len(resultados_df)}")
    print(f"Variáveis significativas (p < 0.05): {len(resultados_df[resultados_df['significancia'] == 'Sim'])}")
    
    # 5. Exportar para Excel
    if output_path is None:
        output_path = 'resultados_anova_medidas_repetidas.xlsx'
    
    print(f"\nSalvando resultados em: {output_path}")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Planilha principal com resultados
        resultados_df.to_excel(writer, sheet_name='Resultados_ANOVA', index=False)
        
        # Planilha com estatísticas descritivas
        descritivas = []
        for variable_name, time_to_col in variable_groups.items():
            if set(time_to_col.keys()) == {'T0', 'T1', 'T2'}:
                cols = [time_to_col['T0'], time_to_col['T1'], time_to_col['T2']]
                stats = df[cols].describe()
                stats.index.name = 'Estatistica'
                stats.columns = ['T0', 'T1', 'T2']
                descritivas.append((variable_name, stats))
        
        # Criar planilhas com estatísticas descritivas
        if descritivas:
            for var_name, stats in descritivas:
                sheet_name = f'Desc_{var_name[:25]}'  # Limitar nome da planilha
                stats.to_excel(writer, sheet_name=sheet_name)
    
    print(f"Análise concluída! Resultados salvos em: {output_path}")
    
    return resultados_df

def main():
    """
    Função principal para executar a análise
    """
    # Caminho para o arquivo CSV
    csv_path = 'analises.csv'
    
    # Verificar se o arquivo existe
    if not Path(csv_path).exists():
        print(f"ERRO: Arquivo não encontrado: {csv_path}")
        print("Por favor, verifique o caminho do arquivo CSV.")
        return
    
    # Executar análise
    resultados = realizar_anova_medidas_repetidas(csv_path)
    
    # Mostrar resultados principais
    print("\n" + "="*80)
    print("RESULTADOS PRINCIPAIS")
    print("="*80)
    
    # Mostrar variáveis significativas
    significativas = resultados[resultados['significancia'] == 'Sim']
    if len(significativas) > 0:
        print(f"\nVariáveis com diferenças significativas (p < 0.05):")
        for _, row in significativas.iterrows():
            print(f"  - {row['Variavel']}: p = {row['p_value']:.4f}, eta2 = {row['partial_eta_squared']:.4f}")
    else:
        print("\nAVISO: Nenhuma variável apresentou diferenças significativas.")
    
    # Mostrar variáveis com maior tamanho de efeito
    grandes_efeitos = resultados[resultados['tamanho_efeito'] == 'Grande']
    if len(grandes_efeitos) > 0:
        print(f"\nVariáveis com grande tamanho de efeito (eta2 >= 0.14):")
        for _, row in grandes_efeitos.iterrows():
            print(f"  - {row['Variavel']}: eta2 = {row['partial_eta_squared']:.4f}")

if __name__ == "__main__":
    main()
