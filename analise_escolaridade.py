import os
import sys
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception as e:
    print("Erro ao importar bibliotecas:\n", e)
    print("Instale dependências: pip install pandas matplotlib seaborn")
    sys.exit(1)


def main(csv_path='base_treino4.csv'):
    df = pd.read_csv(csv_path)

    out_dir = 'outputs'
    plots_dir = 'plots'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 1) Tabela de frequências para Grau de Instrução (coluna 'Escolaridade')
    col = 'Escolaridade'
    freq = df[col].value_counts(dropna=False).rename_axis(col).reset_index(name='Frequencia')
    freq['Percentual'] = (freq['Frequencia'] / freq['Frequencia'].sum() * 100).round(2)
    freq.to_csv(os.path.join(out_dir, 'frequencia_escolaridade.csv'), index=False)
    print('\nTabela de frequências (`Escolaridade`):')
    print(freq.to_string(index=False))

    # 2) Tabela dupla entrada Sexo x Escolaridade com percentuais de linhas e colunas
    ct_counts = pd.crosstab(df['Sexo'], df['Escolaridade'])
    ct_row_pct = ct_counts.div(ct_counts.sum(axis=1), axis=0).multiply(100).round(2)
    ct_col_pct = ct_counts.div(ct_counts.sum(axis=0), axis=1).multiply(100).round(2)

    ct_counts.to_csv(os.path.join(out_dir, 'crosstab_counts.csv'))
    ct_row_pct.to_csv(os.path.join(out_dir, 'crosstab_row_pct.csv'))
    ct_col_pct.to_csv(os.path.join(out_dir, 'crosstab_col_pct.csv'))

    print('\nTabela dupla entrada - Contagens (Sexo x Escolaridade):')
    print(ct_counts)
    print('\nPercentual por linha (row %):')
    print(ct_row_pct)
    print('\nPercentual por coluna (column %):')
    print(ct_col_pct)

    # 3) Representações gráficas
    sns.set(style='whitegrid')

    # Bar plot - distribuição de Escolaridade
    plt.figure(figsize=(8,5))
    ax = sns.countplot(data=df, x=col, order=freq[col].tolist())
    ax.set_title('Distribuição - Escolaridade')
    ax.set_xlabel('Escolaridade')
    ax.set_ylabel('Frequência')
    for p in ax.patches:
        h = int(p.get_height())
        ax.annotate(h, (p.get_x() + p.get_width() / 2., h), ha='center', va='bottom')
    plt.tight_layout()
    bar_path = os.path.join(plots_dir, 'freq_escolaridade_bar.png')
    plt.savefig(bar_path)
    plt.close()

    # Stacked bar plot - Sexo x Escolaridade (proporções)
    ct_norm = ct_counts.div(ct_counts.sum(axis=1), axis=0)
    ct_norm.plot(kind='bar', stacked=True, figsize=(8,6), colormap='tab20')
    plt.title('Proporção de Escolaridade por Sexo (linhas normalizadas)')
    plt.xlabel('Sexo')
    plt.ylabel('Proporção')
    plt.legend(title='Escolaridade', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    stacked_path = os.path.join(plots_dir, 'sexo_escolaridade_stacked.png')
    plt.savefig(stacked_path)
    plt.close()

    # Heatmap de contagens
    plt.figure(figsize=(8,5))
    sns.heatmap(ct_counts, annot=True, fmt='d', cmap='Blues')
    plt.title('Heatmap - Contagens Sexo x Escolaridade')
    plt.tight_layout()
    heatmap_path = os.path.join(plots_dir, 'crosstab_heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()

    print(f"\nArquivos gerados em `{out_dir}/` e `{plots_dir}/`:")
    print(f"- {out_dir}/frequencia_escolaridade.csv")
    print(f"- {out_dir}/crosstab_counts.csv")
    print(f"- {out_dir}/crosstab_row_pct.csv")
    print(f"- {out_dir}/crosstab_col_pct.csv")
    print(f"- {plots_dir}/freq_escolaridade_bar.png")
    print(f"- {plots_dir}/sexo_escolaridade_stacked.png")
    print(f"- {plots_dir}/crosstab_heatmap.png")


if __name__ == '__main__':
    csv = sys.argv[1] if len(sys.argv) > 1 else 'base_treino4.csv'
    if not os.path.exists(csv):
        print(f"Arquivo não encontrado: {csv}")
        sys.exit(1)
    main(csv)
