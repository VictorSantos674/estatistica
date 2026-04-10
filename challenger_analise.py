import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Forçar UTF-8 no stdout (Windows)
sys.stdout.reconfigure(encoding='utf-8')

os.makedirs('outputs', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ==============================================================================
# LEITURA E LIMPEZA DOS DADOS
# ==============================================================================

raw = pd.read_excel('Base_Prova_sala102.xlsx', header=None)

# Localizar onde começa a tabela (linha com "Teste")
header_row = None
for i, row in raw.iterrows():
    if 'Teste' in str(row.values):
        header_row = i
        break

df = pd.read_excel('Base_Prova_sala102.xlsx', header=header_row)
df.columns = ['Teste', 'Temperatura_F', 'Falha', 'Fornecedor']
df = df.dropna(subset=['Teste']).reset_index(drop=True)
df['Teste'] = df['Teste'].astype(int)

# Corrigir encoding do "Não"
df['Falha'] = df['Falha'].astype(str).str.strip()
df['Falha'] = df['Falha'].apply(lambda x: 'Nao' if 'o' in x.lower() and x.lower() != 'sim' else x)
df['Falha'] = df['Falha'].apply(lambda x: 'Sim' if x.lower() == 'sim' else 'Nao')
df['Fornecedor'] = df['Fornecedor'].astype(int)

print("=" * 65)
print("  ANÁLISE CHALLENGER — ANÉIS DE VEDAÇÃO")
print("=" * 65)
print("\nDados brutos (antes da imputação):")
print(df.to_string(index=False))

# ==============================================================================
# PASSO 1 — CONVERTER PARA CSV (antes da imputação)
# ==============================================================================
df.to_csv('outputs/Base_Prova_sala102.csv', index=False)
print("\n[OK] CSV salvo em: outputs/Base_Prova_sala102.csv")

# ==============================================================================
# PASSO 2 — IMPUTAR DADO FALTANTE DA OBSERVAÇÃO 25
# ==============================================================================
# Observação 25: Falha = Sim, Fornecedor = 2 → temperatura ausente
# Estratégia: média das temperaturas dos casos com Falha = Sim

media_falha_sim = df.loc[df['Falha'] == 'Sim', 'Temperatura_F'].mean()
idx_obs25 = df[df['Teste'] == 25].index[0]
df.loc[idx_obs25, 'Temperatura_F'] = round(media_falha_sim, 4)

print(f"\n[Imputação] Observação 25 — temperatura imputada pela média das")
print(f"            temperaturas com Falha = Sim: {media_falha_sim:.4f} °F")
print("\nDados após imputação:")
print(df.to_string(index=False))

df.to_csv('outputs/Base_Prova_sala102_imputado.csv', index=False)
print("\n[OK] CSV com imputação salvo em: outputs/Base_Prova_sala102_imputado.csv")

n_total = len(df)

# ==============================================================================
# QUESTÃO 1 — TABELA DE FREQUÊNCIAS PARA A VARIÁVEL FALHA
# ==============================================================================
print("\n" + "=" * 65)
print("  QUESTÃO 1 — TABELA DE FREQUÊNCIAS: FALHA")
print("=" * 65)

freq_falha = df['Falha'].value_counts().reset_index()
freq_falha.columns = ['Falha', 'Freq_Absoluta']
freq_falha = freq_falha.sort_values('Falha', ascending=False).reset_index(drop=True)
freq_falha['Freq_Relativa'] = freq_falha['Freq_Absoluta'] / n_total
freq_falha['Freq_Relativa_%'] = (freq_falha['Freq_Relativa'] * 100).round(2)

# Adicionar total
total_row = pd.DataFrame([['TOTAL', freq_falha['Freq_Absoluta'].sum(),
                            freq_falha['Freq_Relativa'].sum(),
                            100.0]],
                          columns=freq_falha.columns)
freq_falha_print = pd.concat([freq_falha, total_row], ignore_index=True)

print(freq_falha_print.to_string(index=False))
freq_falha.to_csv('outputs/questao1_tabela_falha.csv', index=False)

# ==============================================================================
# QUESTÃO 2 — TABELA DUPLA ENTRADA: FALHA × FORNECEDOR
# ==============================================================================
print("\n" + "=" * 65)
print("  QUESTÃO 2 — TABELA DUPLA ENTRADA: FALHA × FORNECEDOR")
print("=" * 65)

crosstab = pd.crosstab(df['Falha'], df['Fornecedor'],
                       margins=True, margins_name='Total')
print(crosstab)
crosstab.to_csv('outputs/questao2_tabela_dupla.csv')

# --- Contagens brutas ---
n_falha_sim      = df['Falha'].value_counts()['Sim']          # total Falha = Sim
n_falha_nao      = df['Falha'].value_counts()['Nao']          # total Falha = Nao
n_forn1          = (df['Fornecedor'] == 1).sum()              # total Forn 1
n_forn2          = (df['Fornecedor'] == 2).sum()              # total Forn 2
n_sim_forn1      = ((df['Falha']=='Sim') & (df['Fornecedor']==1)).sum()
n_sim_forn2      = ((df['Falha']=='Sim') & (df['Fornecedor']==2)).sum()
n_nao_forn1      = ((df['Falha']=='Nao') & (df['Fornecedor']==1)).sum()
n_nao_forn2      = ((df['Falha']=='Nao') & (df['Fornecedor']==2)).sum()

# --- 2a) P(Falha) ---
p_falha = n_falha_sim / n_total
print(f"\na) P(Falha) = {n_falha_sim}/{n_total} = {p_falha:.4f}  ({p_falha*100:.2f}%)")

# --- 2b) P(Falha | Fornecedor 1) ---
p_falha_dado_forn1 = n_sim_forn1 / n_forn1
print(f"\nb) P(Falha | Fornecedor 1) = {n_sim_forn1}/{n_forn1} = {p_falha_dado_forn1:.4f}  ({p_falha_dado_forn1*100:.2f}%)")

# --- 2c) P(Falha OU Fornecedor 1) = P(F) + P(F1) - P(F e F1) ---
p_forn1         = n_forn1 / n_total
p_falha_e_forn1 = n_sim_forn1 / n_total
p_falha_ou_forn1 = p_falha + p_forn1 - p_falha_e_forn1
print(f"\nc) P(Falha ou Fornecedor 1)")
print(f"   = P(Falha) + P(Forn1) - P(Falha n Forn1)")
print(f"   = {p_falha:.4f} + {p_forn1:.4f} - {p_falha_e_forn1:.4f}")
print(f"   = {p_falha_ou_forn1:.4f}  ({p_falha_ou_forn1*100:.2f}%)")

# Salvar probabilidades
probs_df = pd.DataFrame({
    'Probabilidade': ['P(Falha)', 'P(Falha | Forn1)', 'P(Falha ou Forn1)'],
    'Valor': [round(p_falha,4), round(p_falha_dado_forn1,4), round(p_falha_ou_forn1,4)],
    'Percentual': [f'{p_falha*100:.2f}%',
                   f'{p_falha_dado_forn1*100:.2f}%',
                   f'{p_falha_ou_forn1*100:.2f}%']
})
probs_df.to_csv('outputs/questao2_probabilidades.csv', index=False)

# ==============================================================================
# QUESTÃO 3 — FALHA TEM RELAÇÃO COM FORNECEDOR?
# ==============================================================================
print("\n" + "=" * 65)
print("  QUESTÃO 3 — FALHA × FORNECEDOR (análise)")
print("=" * 65)

p_falha_forn2 = n_sim_forn2 / n_forn2
print(f"\n  P(Falha | Fornecedor 1) = {n_sim_forn1}/{n_forn1} = {p_falha_dado_forn1:.4f} ({p_falha_dado_forn1*100:.2f}%)")
print(f"  P(Falha | Fornecedor 2) = {n_sim_forn2}/{n_forn2} = {p_falha_forn2:.4f} ({p_falha_forn2*100:.2f}%)")
print(f"  P(Falha) geral          = {p_falha:.4f} ({p_falha*100:.2f}%)")

# ==============================================================================
# QUESTÃO 4 — GRÁFICO: TEMPERATURA SEGMENTADA POR FALHA
# ==============================================================================
print("\n" + "=" * 65)
print("  QUESTÃO 4 — GRÁFICO: TEMPERATURA SEGMENTADA POR FALHA")
print("=" * 65)

cores = {'Sim': '#e74c3c', 'Nao': '#2ecc71'}
labels_map = {'Sim': 'Falha = Sim', 'Nao': 'Falha = Não'}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Temperatura (°F) Segmentada por Falha — Anéis Challenger',
             fontsize=14, fontweight='bold', y=1.01)

# ---- Boxplot ----
grupos = [df.loc[df['Falha']=='Sim','Temperatura_F'].values,
          df.loc[df['Falha']=='Nao','Temperatura_F'].values]

bp = axes[0].boxplot(grupos, patch_artist=True, widths=0.5,
                     medianprops=dict(color='black', linewidth=2))
bp['boxes'][0].set_facecolor('#e74c3c')
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor('#2ecc71')
bp['boxes'][1].set_alpha(0.7)

axes[0].set_xticks([1, 2])
axes[0].set_xticklabels(['Falha = Sim', 'Falha = Não'], fontsize=12)
axes[0].set_ylabel('Temperatura (°F)', fontsize=12)
axes[0].set_title('Boxplot', fontsize=12)
axes[0].axhline(y=24, color='navy', linestyle='--', linewidth=1.5,
                label='Temp. acidente (24 °F)')
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.4)

# Anotar mediana
for i, grp in enumerate(grupos, 1):
    med = np.median(grp)
    axes[0].text(i + 0.15, med, f'{med:.1f}', va='center', fontsize=9, color='black')

# ---- Dispersão (stripplot manual) ----
for cat, cor in cores.items():
    sub = df[df['Falha'] == cat]
    jitter = np.random.uniform(-0.05, 0.05, len(sub))
    x_pos = (1 if cat == 'Sim' else 2) + jitter
    axes[1].scatter(x_pos, sub['Temperatura_F'], color=cor, alpha=0.75,
                    edgecolors='black', linewidths=0.5, s=60,
                    label=labels_map[cat])

axes[1].axhline(y=24, color='navy', linestyle='--', linewidth=1.8,
                label='Temp. acidente (24 °F)')
axes[1].set_xticks([1, 2])
axes[1].set_xticklabels(['Falha = Sim', 'Falha = Não'], fontsize=12)
axes[1].set_ylabel('Temperatura (°F)', fontsize=12)
axes[1].set_title('Dispersão individual', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.4)

plt.tight_layout()
plt.savefig('plots/questao4_temperatura_por_falha.png', dpi=200, bbox_inches='tight')
plt.close()
print("[OK] Gráfico salvo em: plots/questao4_temperatura_por_falha.png")

# ==============================================================================
# QUESTÃO 5 — 24 °F PODE TER CAUSADO O ACIDENTE?
# ==============================================================================
print("\n" + "=" * 65)
print("  QUESTÃO 5 — ANÁLISE: 24 °F E O ACIDENTE")
print("=" * 65)

temp_acidente = 24
temp_min_experimento = df['Temperatura_F'].min()
temp_max_falha = df.loc[df['Falha']=='Sim','Temperatura_F'].max()
temp_min_falha = df.loc[df['Falha']=='Sim','Temperatura_F'].min()
temp_min_ok    = df.loc[df['Falha']=='Nao','Temperatura_F'].min()
media_sim      = df.loc[df['Falha']=='Sim','Temperatura_F'].mean()
media_nao      = df.loc[df['Falha']=='Nao','Temperatura_F'].mean()

print(f"\n  Temperatura no acidente  : {temp_acidente} °F")
print(f"  Mínimo nos experimentos  : {temp_min_experimento:.2f} °F")
print(f"  Máx. temperatura c/ Falha: {temp_max_falha:.2f} °F")
print(f"  Mín. temperatura c/ Falha: {temp_min_falha:.2f} °F")
print(f"  Mín. temperatura sem Falha: {temp_min_ok:.2f} °F")
print(f"  Média temperatura c/ Falha: {media_sim:.2f} °F")
print(f"  Média temperatura sem Falha: {media_nao:.2f} °F")

# ==============================================================================
# RESUMO FINAL
# ==============================================================================
print("\n" + "=" * 65)
print("  RESUMO DE ARQUIVOS GERADOS")
print("=" * 65)
print("  outputs/Base_Prova_sala102.csv")
print("  outputs/Base_Prova_sala102_imputado.csv")
print("  outputs/questao1_tabela_falha.csv")
print("  outputs/questao2_tabela_dupla.csv")
print("  outputs/questao2_probabilidades.csv")
print("  plots/questao4_temperatura_por_falha.png")
print("=" * 65)
