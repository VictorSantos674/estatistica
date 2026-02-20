import pandas as pd

# Ler o arquivo CSV
df = pd.read_csv("base_treino4.csv")

# Criar dicionário de mapeamento
mapa_midia = {
    1: "TV",
    2: "Internet",
    3: "Site de Compra"
}

# Criar nova coluna com os nomes das mídias
df["Midia_Nome"] = df["Midia"].map(mapa_midia)

# Calcular frequência
frequencia = df["Midia_Nome"].value_counts()

# Descobrir a mídia mais utilizada
midia_mais_utilizada = frequencia.idxmax()
quantidade = frequencia.max()

print("Frequência por mídia:")
print(frequencia)
print("\nMeio de mídia mais utilizado:")
print(f"{midia_mais_utilizada} com {quantidade} ocorrências")