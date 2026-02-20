import pandas as pd

# Ler o arquivo CSV
df = pd.read_csv("base_treino4.csv")

# Frequência Absoluta
freq_abs = df['Sexo'].value_counts().sort_index()

# Frequência Relativa
freq_rel = df['Sexo'].value_counts(normalize=True).sort_index()

# Frequência Percentual
freq_perc = freq_rel * 100

# Frequência Acumulada
freq_acum = freq_abs.cumsum()

# Montando a Tabela de Distribuição de Frequências
tabela_frequencia = pd.DataFrame({
    'Frequência Absoluta (fi)': freq_abs,
    'Frequência Relativa (fr)': freq_rel,
    'Percentual (%)': freq_perc,
    'Frequência Acumulada (Fi)': freq_acum
})

# Resetando índice para virar coluna
tabela_frequencia = tabela_frequencia.reset_index()
tabela_frequencia = tabela_frequencia.rename(columns={'index': 'Sexo'})

print(tabela_frequencia)