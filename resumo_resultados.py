import pandas as pd

# Ler os resultados
df_anova = pd.read_excel('analise_completa_todas_variaveis.xlsx', sheet_name='ANOVA')
df_posthoc = pd.read_excel('analise_completa_todas_variaveis.xlsx', sheet_name='PostHoc')

print('RESUMO DOS RESULTADOS SIGNIFICATIVOS')
print('='*50)
print(f'Total de variáveis analisadas: {len(df_anova)}')

# Filtrar resultados significativos
significativos = df_anova[df_anova['significativo'] == 'Sim']
print(f'Variáveis com ANOVA significativa: {len(significativos)}')
print()

print('VARIÁVEIS COM DIFERENÇAS SIGNIFICATIVAS:')
print('-'*50)
for _, row in significativos.iterrows():
    print(f'{row["Variavel"]}: F={row["F"]:.3f}, p={row["p_value"]:.4f}, η²={row["partial_eta_squared"]:.4f} ({row["tamanho_efeito"]})')

print()
print('COMPARAÇÕES POST-HOC SIGNIFICATIVAS:')
print('-'*50)
posthoc_significativos = df_posthoc[df_posthoc['Significativo'] == 'Sim']
print(f'Total de comparações post-hoc: {len(df_posthoc)}')
print(f'Comparações significativas: {len(posthoc_significativos)}')
print()

for _, row in posthoc_significativos.iterrows():
    print(f'{row["Variavel"]} - {row["Comparacao"]}: p={row["P_corrigido"]:.3f}, Cohen\'s d={row["Tamanho_efeito"]:.3f}')
