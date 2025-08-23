# Análise de Dados do Teste Sternberg

Este projeto contém scripts para análise de dados do teste Sternberg, um paradigma experimental clássico em psicologia cognitiva que mede a memória de trabalho e o tempo de resposta.

## 📁 Estrutura do Projeto

```
test-sternber/
├── data/                           # Dados brutos organizados por participante e momento
│   ├── T0_4567_sternberg.csv      # Teste inicial (T0) para participante 4567
│   ├── T1_4567_sternberg.csv      # Teste intermediário (T1) para participante 4567
│   ├── T2_4567_sternberg.csv      # Teste final (T2) para participante 4567
│   └── ...                        # Arquivos para outros participantes
├── dados_sternberg_combinados/     # Dados consolidados por participante
│   ├── 4567_sternberg_combined.csv # Dados T0, T1 e T2 combinados para participante 4567
│   └── ...                        # Arquivos combinados para outros participantes
├── combine_sternberg_data.py       # Script para combinar dados dos três momentos
├── analises.py                     # Script para calcular métricas e estatísticas
├── anova.py                        # Script para análise estatística ANOVA
├── analises.csv                    # Arquivo de saída com métricas calculadas
└── resultados_anova_medidas_repetidas.xlsx # Resultados da análise estatística
```

## 🔬 Descrição dos Scripts

### 1. `combine_sternberg_data.py` - Consolidação de Dados

**Propósito**: Combina os dados dos três momentos de teste (T0, T1, T2) para cada participante em um único arquivo.

**Funcionalidades**:
- **Extração de IDs**: Identifica automaticamente o ID do participante e número do teste a partir do nome do arquivo
- **Consolidação horizontal**: Combina as colunas dos três testes lado a lado
- **Prefixação**: Adiciona prefixos T0_, T1_, T2_ às colunas para distinguir os momentos
- **Conversão de tipos**: Converte colunas numéricas para o tipo apropriado
- **Validação**: Verifica se todos os três testes estão presentes para cada participante

**Entrada**: Arquivos CSV individuais em `data/` (T0_*, T1_*, T2_*)
**Saída**: Arquivos consolidados em `dados_sternberg_combinados/` com formato:
```
ID,Tamanho,Trial,Conjunto,Estímulo,Verdadeiro/Falso,Resposta,Correto,Tempo(ms),ID,Tamanho,Trial,Conjunto,Estímulo,Verdadeiro/Falso,Resposta,Correto,Tempo(ms),ID,Tamanho,Trial,Conjunto,Estímulo,Verdadeiro/Falso,Resposta,Correto,Tempo(ms)
T0_subNum,T0_length,T0_trial,T0_set,T0_stim,T0_targetfoil,T0_resp,T0_corr,T0_rt,T1_subNum,T1_length,T1_trial,T1_set,T1_stim,T1_targetfoil,T1_resp,T1_corr,T1_rt,T2_subNum,T2_length,T2_trial,T2_set,T2_stim,T2_targetfoil,T2_resp,T2_corr,T2_rt
```

### 2. `analises.py` - Cálculo de Métricas

**Propósito**: Calcula métricas detalhadas de performance para cada participante nos três momentos de teste.

**Métricas Calculadas**:

#### **Tempo de Resposta (RT)**
- `mean_rt_total_T0/T1/T2`: RT médio total para cada momento
- `mean_rt_by_length_T0/T1/T2_2/4/6`: RT médio por tamanho do conjunto (2, 4, 6 itens)
- `mean_rt_correct_T0/T1/T2`: RT médio apenas para respostas corretas
- `mean_rt_incorrect_T0/T1/T2`: RT médio apenas para respostas incorretas
- `mean_rt_correct_by_length_T0/T1/T2_2/4/6`: RT médio para respostas corretas por tamanho
- `mean_rt_incorrect_by_length_T0/T1/T2_2/4/6`: RT médio para respostas incorretas por tamanho

#### **Precisão (Accuracy)**
- `accuracy_total_T0/T1/T2`: Proporção total de respostas corretas
- `accuracy_by_length_T0/T1/T2_2/4/6`: Precisão por tamanho do conjunto
- `accuracy_target_T0/T1/T2`: Precisão para estímulos alvo (T)
- `accuracy_foil_T0/T1/T2`: Precisão para estímulos distrator (F)
- `accuracy_target_by_length_T0/T1/T2_2/4/6`: Precisão para alvos por tamanho
- `accuracy_foil_by_length_T0/T1/T2_2/4/6`: Precisão para distratores por tamanho

#### **Slope de Memória**
- `slope_rt_by_length_T0/T1/T2`: Inclinação da regressão RT vs. tamanho do conjunto (ms/item)

**Entrada**: Arquivos consolidados de `dados_sternberg_combinados/`
**Saída**: `analises.csv` com uma linha por participante e todas as métricas calculadas

### 3. `anova.py` - Análise Estatística

**Propósito**: Realiza análise de variância (ANOVA) de medidas repetidas para identificar diferenças significativas entre os três momentos de teste.

**Funcionalidades**:
- **Detecção automática**: Identifica automaticamente variáveis com padrão T0/T1/T2
- **ANOVA de medidas repetidas**: Usa a biblioteca `pingouin` para análises estatísticas robustas
- **Múltiplas variáveis**: Analisa todas as variáveis disponíveis simultaneamente
- **Estatísticas completas**: Calcula F-value, p-value, partial eta squared e tamanho do efeito
- **Exportação Excel**: Gera relatório completo com múltiplas planilhas

**Métricas Estatísticas**:
- **F-value**: Estatística F da ANOVA
- **p-value**: Significância estatística (α = 0.05)
- **Partial eta squared**: Tamanho do efeito (pequeno: <0.06, médio: 0.06-0.14, grande: >0.14)
- **Significância**: Classificação binária (Sim/Não) baseada no p-value

**Entrada**: `analises.csv` gerado pelo script anterior
**Saída**: `resultados_anova_medidas_repetidas.xlsx` com:
- Planilha principal com resultados das ANOVAs
- Planilhas separadas com estatísticas descritivas para cada variável

## 📊 Estrutura dos Dados

### Dados Brutos (pasta `data/`)
Cada arquivo contém dados de um participante em um momento específico:
- **subNum**: Identificador do participante
- **length**: Tamanho do conjunto de memória (2, 4, 6 itens)
- **trial**: Número do trial
- **set**: Conjunto de letras apresentado
- **stim**: Estímulo testado
- **targetfoil**: Tipo de estímulo (T=alvo, F=distrator)
- **resp**: Resposta do participante
- **corr**: Correção da resposta (1=correto, 0=incorreto)
- **rt**: Tempo de resposta em milissegundos

### Dados Consolidados (pasta `dados_sternberg_combinados/`)
Arquivos com dados dos três momentos (T0, T1, T2) combinados horizontalmente, com prefixos para distinguir os momentos.

### Métricas Calculadas (`analises.csv`)
Arquivo com uma linha por participante contendo todas as métricas calculadas para os três momentos de teste.

## 🚀 Como Usar

### 1. Preparação dos Dados
```bash
python combine_sternberg_data.py
```
Este comando:
- Lê todos os arquivos CSV da pasta `data/`
- Combina os dados T0, T1 e T2 para cada participante
- Salva arquivos consolidados em `dados_sternberg_combinados/`

### 2. Cálculo das Métricas
```bash
python analises.py
```
Este comando:
- Processa todos os arquivos consolidados
- Calcula métricas de RT, precisão e slope
- Gera o arquivo `analises.csv`

### 3. Análise Estatística
```bash
python anova.py
```
Este comando:
- Lê o arquivo `analises.csv`
- Realiza ANOVAs de medidas repetidas para todas as variáveis
- Gera relatório Excel com resultados estatísticos

## 📋 Dependências

- **pandas**: Manipulação e análise de dados
- **numpy**: Operações numéricas
- **pingouin**: Análises estatísticas avançadas
- **openpyxl**: Exportação para Excel
- **pathlib**: Manipulação de caminhos de arquivo

## 🔍 Interpretação dos Resultados

### Métricas de Performance
- **RT médio**: Tempo de resposta - valores menores indicam melhor performance
- **Precisão**: Proporção de respostas corretas - valores próximos a 1.0 indicam melhor performance
- **Slope**: Inclinação da regressão RT vs. tamanho - valores menores indicam melhor eficiência de memória

### Significância Estatística
- **p < 0.05**: Diferença significativa entre os momentos de teste
- **Tamanho do efeito**: Indica a magnitude prática da diferença
- **F-value**: Estatística de teste - valores maiores indicam diferenças mais pronunciadas

## 📝 Notas Importantes

1. **Estrutura de arquivos**: Os scripts esperam uma estrutura específica de pastas e nomenclatura
2. **Valores ausentes**: Os scripts tratam automaticamente valores NaN e dados inválidos
3. **Logs detalhados**: Cada script fornece informações sobre o processamento
4. **Validação de dados**: Os scripts verificam a integridade dos dados antes do processamento
5. **Formato de saída**: Os resultados são salvos em formatos padrão (CSV e Excel) para fácil análise posterior

## 🤝 Contribuições

Este projeto foi desenvolvido para análise de dados experimentais do teste Sternberg. Para dúvidas ou sugestões, consulte a documentação dos scripts individuais ou entre em contato com a equipe de desenvolvimento.
