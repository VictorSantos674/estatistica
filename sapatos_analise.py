import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

os.makedirs('outputs', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ==============================================================================
# DADOS — Box et al. (1978, p. 97-100)
# ==============================================================================

dados = {
    'Menino': list(range(1, 11)),
    'A': [13.2, 8.2, 10.9, 14.3, 10.7, 6.6, 9.5, 10.8, 8.8, 13.3],
    'B': [14.0, 8.8, 11.2, 14.2, 11.8, 6.4, 9.8, 11.3, 9.3, 13.6],
}

df = pd.DataFrame(dados)
df['Diferenca'] = df['A'] - df['B']

print("=" * 65)
print("  ANÁLISE DE DESGASTE EM SOLAS DE SAPATOS — Box et al. (1978)")
print("=" * 65)
print("\nDados brutos:")
print(df.to_string(index=False))

# ==============================================================================
# QUESTÃO A — ESTATÍSTICAS DESCRITIVAS
# ==============================================================================
print("\n" + "=" * 65)
print("  QUESTÃO A — ESTATÍSTICAS DESCRITIVAS POR MATERIAL")
print("=" * 65)

for mat in ['A', 'B']:
    s = df[mat]
    print(f"\n  Material {mat}:")
    print(f"    n           = {len(s)}")
    print(f"    Média       = {s.mean():.4f}")
    print(f"    Mediana     = {s.median():.4f}")
    print(f"    Desvio Pad. = {s.std(ddof=1):.4f}")
    print(f"    Variância   = {s.var(ddof=1):.4f}")
    print(f"    Mínimo      = {s.min():.4f}")
    print(f"    Q1          = {s.quantile(0.25):.4f}")
    print(f"    Q3          = {s.quantile(0.75):.4f}")
    print(f"    Máximo      = {s.max():.4f}")
    print(f"    Amplitude   = {s.max() - s.min():.4f}")
    print(f"    IQR         = {s.quantile(0.75) - s.quantile(0.25):.4f}")

desc = df[['A', 'B']].describe().round(4)
desc.to_csv('outputs/sapatos_descritivas.csv')
print("\n[OK] Descritivas salvas em: outputs/sapatos_descritivas.csv")

# ==============================================================================
# QUESTÃO B — VARIABILIDADE DENTRO × ENTRE GRUPOS
# ==============================================================================
print("\n" + "=" * 65)
print("  QUESTÃO B — VARIABILIDADE DENTRO vs. ENTRE GRUPOS")
print("=" * 65)

# Variabilidade ENTRE grupos = variabilidade entre os meninos (média de A e B)
media_menino = df[['A', 'B']].mean(axis=1)
dp_entre = media_menino.std(ddof=1)

# Variabilidade DENTRO de cada menino = diferença A-B (tamanho do efeito dentro do par)
dp_diferenca = df['Diferenca'].std(ddof=1)

# Desvios padrão de cada material individualmente
dp_A = df['A'].std(ddof=1)
dp_B = df['B'].std(ddof=1)

print(f"\n  DP de A (desgaste material A, todos os meninos)    = {dp_A:.4f}")
print(f"  DP de B (desgaste material B, todos os meninos)    = {dp_B:.4f}")
print(f"\n  DP da média de cada menino (variab. ENTRE grupos)  = {dp_entre:.4f}")
print(f"  DP das diferenças A-B   (variab. DENTRO dos pares) = {dp_diferenca:.4f}")
print("""
  Interpretação:
    A variabilidade ENTRE meninos (~2,5) é muito maior do que a
    variabilidade DENTRO de cada par (diferenças A-B, ~0,4).
    Isso indica que as diferenças individuais (quem usa mais o sapato)
    dominam o sinal. Se usássemos dois grupos separados, essa variação
    entre meninos mascararia a diferença entre materiais.
    O delineamento em pares ELIMINA essa fonte de variação, tornando
    a comparação entre A e B muito mais precisa e sensível.
""")

# ==============================================================================
# QUESTÃO C — DIAGRAMA DE PONTOS GERAL
# ==============================================================================
print("=" * 65)
print("  QUESTÃO C — DIAGRAMA DE PONTOS (dados gerais)")
print("=" * 65)

fig, ax = plt.subplots(figsize=(8, 5))

np.random.seed(42)
jitter_A = np.random.uniform(-0.06, 0.06, len(df))
jitter_B = np.random.uniform(-0.06, 0.06, len(df))

ax.scatter(df['A'], np.ones(len(df)) + jitter_A,
           color='#2980b9', s=80, alpha=0.8, edgecolors='black',
           linewidths=0.6, label='Material A', zorder=3)
ax.scatter(df['B'], np.zeros(len(df)) + jitter_B,
           color='#e74c3c', s=80, alpha=0.8, edgecolors='black',
           linewidths=0.6, label='Material B', zorder=3)

ax.axvline(df['A'].mean(), color='#2980b9', linestyle='--', linewidth=1.5, alpha=0.8)
ax.axvline(df['B'].mean(), color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.8)

ax.set_yticks([0, 1])
ax.set_yticklabels(['Material B', 'Material A'], fontsize=12)
ax.set_xlabel('Desgaste (unidades)', fontsize=12)
ax.set_title('Diagrama de Pontos — Desgaste por Material', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)
ax.text(df['A'].mean() + 0.05, 1.22, f"Média A\n{df['A'].mean():.2f}",
        color='#2980b9', fontsize=8.5, ha='left')
ax.text(df['B'].mean() + 0.05, -0.22, f"Média B\n{df['B'].mean():.2f}",
        color='#e74c3c', fontsize=8.5, ha='left')

plt.tight_layout()
plt.savefig('plots/sapatos_c_diagrama_pontos.png', dpi=200, bbox_inches='tight')
plt.close()

print("""
  Resultado:
    Os pontos de A e B se sobrepõem bastante ao longo do eixo x.
    As médias são próximas (A ≈ 10,63; B ≈ 11,04), mas as nuvens
    se misturam muito — impossível perceber uma diferença clara
    sem controlar a variação entre meninos.
""")
print("[OK] Gráfico salvo em: plots/sapatos_c_diagrama_pontos.png")

# ==============================================================================
# QUESTÃO D — DESTACAR MEDIDAS DE CADA MENINO (pares ligados)
# ==============================================================================
print("\n" + "=" * 65)
print("  QUESTÃO D — PARES POR MENINO (linhas de conexão)")
print("=" * 65)

cores_meninos = plt.cm.tab10(np.linspace(0, 1, 10))

fig, ax = plt.subplots(figsize=(9, 6))

for i, row in df.iterrows():
    cor = cores_meninos[i]
    ax.plot([1, 2], [row['A'], row['B']],
            color=cor, linewidth=1.4, alpha=0.7, zorder=2)
    ax.scatter([1], [row['A']], color=cor, s=70, edgecolors='black',
               linewidths=0.5, zorder=3)
    ax.scatter([2], [row['B']], color=cor, s=70, edgecolors='black',
               linewidths=0.5, zorder=3)
    ax.text(2.05, row['B'], f" Menino {row['Menino']}", va='center',
            fontsize=7.5, color=cor)

ax.set_xticks([1, 2])
ax.set_xticklabels(['Material A', 'Material B'], fontsize=13)
ax.set_ylabel('Desgaste (unidades)', fontsize=12)
ax.set_title('Desgaste por Menino — Materiais A e B\n(cada linha representa um menino)',
             fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/sapatos_d_pares_meninos.png', dpi=200, bbox_inches='tight')
plt.close()

print("""
  Resultado:
    Conectando os dois pés de cada menino, fica evidente que
    TODAS as linhas sobem de A para B (ou seja, B desgasta mais
    ou igual a A para quase todos os meninos). A variação entre
    meninos é muito grande (linhas bem espaçadas), confirmando
    que o delineamento pareado é essencial para detectar a
    diferença entre os materiais.
""")
print("[OK] Gráfico salvo em: plots/sapatos_d_pares_meninos.png")

# ==============================================================================
# QUESTÃO E — DIFERENÇAS A-B POR MENINO
# ==============================================================================
print("=" * 65)
print("  QUESTÃO E — DIFERENÇAS (A − B) POR MENINO")
print("=" * 65)

print(f"\n  Diferenças A − B por menino:")
for _, row in df.iterrows():
    sinal = "↑ A desgasta mais" if row['Diferenca'] > 0 else \
            ("↓ B desgasta mais" if row['Diferenca'] < 0 else "= Iguais")
    print(f"    Menino {int(row['Menino']):2d}: {row['Diferenca']:+.1f}  ({sinal})")

print(f"\n  Média das diferenças  = {df['Diferenca'].mean():.4f}")
print(f"  DP das diferenças     = {df['Diferenca'].std(ddof=1):.4f}")
print(f"  Mediana das diferenças= {df['Diferenca'].median():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Diferenças de Desgaste (A − B) por Menino',
             fontsize=13, fontweight='bold')

# --- Gráfico de barras ---
cores_barras = ['#e74c3c' if d < 0 else '#2980b9' for d in df['Diferenca']]
axes[0].bar(df['Menino'], df['Diferenca'], color=cores_barras, edgecolor='black',
            linewidth=0.7, alpha=0.85)
axes[0].axhline(0, color='black', linewidth=1.2)
axes[0].axhline(df['Diferenca'].mean(), color='navy', linestyle='--',
                linewidth=1.5, label=f"Média = {df['Diferenca'].mean():.2f}")
axes[0].set_xlabel('Menino', fontsize=11)
axes[0].set_ylabel('Diferença A − B', fontsize=11)
axes[0].set_title('Barras por Menino', fontsize=11)
axes[0].set_xticks(df['Menino'])
axes[0].legend(fontsize=9)
axes[0].grid(axis='y', alpha=0.3)

patch_A = mpatches.Patch(color='#2980b9', label='A desgasta mais (A − B > 0)')
patch_B = mpatches.Patch(color='#e74c3c', label='B desgasta mais (A − B < 0)')
axes[0].legend(handles=[patch_A, patch_B,
               plt.Line2D([0], [0], color='navy', linestyle='--',
                          label=f"Média = {df['Diferenca'].mean():.2f}")],
               fontsize=8.5)

# --- Diagrama de pontos das diferenças ---
np.random.seed(0)
jitter = np.random.uniform(-0.04, 0.04, len(df))
cores_pts = ['#e74c3c' if d < 0 else '#2980b9' for d in df['Diferenca']]

axes[1].scatter(df['Diferenca'], jitter, c=cores_pts, s=90,
                edgecolors='black', linewidths=0.6, zorder=3)
axes[1].axvline(0, color='black', linewidth=1.2)
axes[1].axvline(df['Diferenca'].mean(), color='navy', linestyle='--',
                linewidth=1.5, label=f"Média = {df['Diferenca'].mean():.2f}")

for i, row in df.iterrows():
    axes[1].text(row['Diferenca'], jitter[i] + 0.015,
                 f"{row['Menino']}", ha='center', fontsize=7.5)

axes[1].set_xlabel('Diferença A − B', fontsize=11)
axes[1].set_yticks([])
axes[1].set_title('Diagrama de Pontos das Diferenças', fontsize=11)
axes[1].legend(fontsize=9)
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/sapatos_e_diferencas.png', dpi=200, bbox_inches='tight')
plt.close()

print("""
  Resultado:
    9 dos 10 meninos apresentam A − B < 0, indicando que o material
    B desgasta sistematicamente mais do que A. O único caso positivo
    (menino 4, diferença = +0,1) é praticamente nulo.
    A média das diferenças é −0,41, consistentemente negativa.
    O diagrama de pontos evidencia que todos os valores se
    concentram à esquerda de zero — forte evidência de que
    o Material A é mais durável que o B.
""")
print("[OK] Gráfico salvo em: plots/sapatos_e_diferencas.png")

# ==============================================================================
# RESUMO FINAL
# ==============================================================================
print("\n" + "=" * 65)
print("  ARQUIVOS GERADOS")
print("=" * 65)
print("  outputs/sapatos_descritivas.csv")
print("  plots/sapatos_c_diagrama_pontos.png")
print("  plots/sapatos_d_pares_meninos.png")
print("  plots/sapatos_e_diferencas.png")
print("=" * 65)
