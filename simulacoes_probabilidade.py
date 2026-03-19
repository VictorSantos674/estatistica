import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo dos gráficos
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Criar diretórios de saída
os.makedirs('outputs', exist_ok=True)
os.makedirs('plots', exist_ok=True)


def questao_1_moeda():
    """
    Questão 1: Simular 50 lançamentos de moeda (0=cara, 1=coroa)
    e depois recodificar para cara e coroa
    """
    print("\n" + "="*70)
    print("QUESTÃO 1: Simulação de Moeda (50 lançamentos)")
    print("="*70)
    
    np.random.seed(42)
    # Simular 50 lançamentos (0=cara, 1=coroa)
    lancamentos = np.random.binomial(n=1, p=0.5, size=50)
    
    # Recodificar para cara e coroa
    recodificado = np.array(['Cara' if x == 0 else 'Coroa' for x in lancamentos])
    
    print("\nPrimeiros 20 lançamentos (numérico):")
    print(lancamentos[:20])
    print("\nPrimeiros 20 lançamentos (recodificado):")
    print(recodificado[:20])
    
    # Salvar sequência em arquivo
    df_seq = pd.DataFrame({
        'Lançamento': range(1, 51),
        'Valor_Numérico': lancamentos,
        'Resultado': recodificado
    })
    df_seq.to_csv('outputs/questao1_sequencia_moeda.csv', index=False)
    print("\nSequência salva em: outputs/questao1_sequencia_moeda.csv")
    
    return lancamentos, recodificado


def questao_2_moeda(lancamentos, recodificado):
    """
    Questão 2: Tabela de frequências absoluta e relativa
    Calcular diferenças em relação aos valores esperados
    """
    print("\n" + "="*70)
    print("QUESTÃO 2: Tabela de Frequências - Moeda")
    print("="*70)
    
    # Frequências absolutas
    valores_unicos, contagens = np.unique(recodificado, return_counts=True)
    freq_abs = pd.DataFrame({
        'Resultado': valores_unicos,
        'Frequência_Absoluta': contagens
    })
    
    # Frequências relativas
    freq_abs['Frequência_Relativa'] = (freq_abs['Frequência_Absoluta'] / len(recodificado)).round(4)
    
    # Valores esperados (moeda honesta: 0.5 para cada)
    freq_abs['Frequência_Esperada'] = 0.5
    
    # Diferença entre observado e esperado
    freq_abs['Diferença'] = (freq_abs['Frequência_Relativa'] - freq_abs['Frequência_Esperada']).round(4)
    
    print("\n", freq_abs.to_string(index=False))
    freq_abs.to_csv('outputs/questao2_frequencias_moeda.csv', index=False)
    print("\nTabela salva em: outputs/questao2_frequencias_moeda.csv")
    
    return freq_abs


def questao_3_dado():
    """
    Questão 3: Simular 80 lançamentos de dado
    a) Distribuição de frequências (absoluta e relativa)
    b) Média e variância
    c) Gráfico de frequências relativas
    d) Gráfico comparando observado vs esperado
    """
    print("\n" + "="*70)
    print("QUESTÃO 3: Simulação de Dado (80 lançamentos)")
    print("="*70)
    
    np.random.seed(42)
    # Simular 80 lançamentos de dado (valores 1-6)
    lancamentos = np.random.randint(1, 7, size=80)
    
    # Frequências absolutas
    valores_unicos = np.arange(1, 7)
    contagens = np.array([np.sum(lancamentos == i) for i in valores_unicos])
    
    freq_df = pd.DataFrame({
        'Face': valores_unicos,
        'Frequência_Absoluta': contagens,
        'Frequência_Relativa': (contagens / len(lancamentos)).round(4),
        'Frequência_Esperada': 1/6  # Para dado honesto
    })
    
    # Diferença observado - esperado
    freq_df['Diferença'] = (freq_df['Frequência_Relativa'] - freq_df['Frequência_Esperada']).round(4)
    
    print("\n3a) Distribuição de Frequências:")
    print(freq_df.to_string(index=False))
    freq_df.to_csv('outputs/questao3a_frequencias_dado.csv', index=False)
    
    # 3b) Média e Variância
    media = np.mean(lancamentos)
    variancia = np.var(lancamentos, ddof=1)  # ddof=1 para variância amostral
    desvio_padrao = np.std(lancamentos, ddof=1)
    
    print(f"\n3b) Estatísticas dos Lançamentos:")
    print(f"    Média: {media:.4f}")
    print(f"    Variância: {variancia:.4f}")
    print(f"    Desvio Padrão: {desvio_padrao:.4f}")
    
    stats_df = pd.DataFrame({
        'Estatística': ['Média', 'Variância', 'Desvio Padrão'],
        'Valor_Observado': [media, variancia, desvio_padrao],
        'Valor_Esperado': [3.5, 35/12, np.sqrt(35/12)]
    })
    stats_df['Diferença'] = (stats_df['Valor_Observado'] - stats_df['Valor_Esperado']).round(4)
    print("\n", stats_df.to_string(index=False))
    stats_df.to_csv('outputs/questao3b_estatisticas_dado.csv', index=False)
    
    # 3c) Gráfico de frequências relativas
    plt.figure(figsize=(10, 6))
    plt.bar(freq_df['Face'], freq_df['Frequência_Relativa'], color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Face do Dado', fontsize=12)
    plt.ylabel('Frequência Relativa', fontsize=12)
    plt.title('Distribuição de Frequências Relativas - Dado (80 lançamentos)', fontsize=13, fontweight='bold')
    plt.xticks(valores_unicos)
    plt.ylim(0, max(freq_df['Frequência_Relativa']) * 1.2)
    for i, (face, freq) in enumerate(zip(freq_df['Face'], freq_df['Frequência_Relativa'])):
        plt.text(face, freq + 0.01, f'{freq:.2%}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('plots/questao3c_frequencias_dado_barras.png', dpi=300)
    plt.close()
    print("\nGráfico salvo em: plots/questao3c_frequencias_dado_barras.png")
    
    # 3d) Gráfico comparando observado vs esperado
    plt.figure(figsize=(10, 6))
    x = np.arange(len(freq_df['Face']))
    width = 0.35
    
    plt.bar(x - width/2, freq_df['Frequência_Relativa'], width, label='Observado', 
            color='skyblue', edgecolor='black', alpha=0.7)
    plt.bar(x + width/2, freq_df['Frequência_Esperada'], width, label='Esperado (honesto)', 
            color='salmon', edgecolor='black', alpha=0.7)
    
    plt.xlabel('Face do Dado', fontsize=12)
    plt.ylabel('Frequência Relativa', fontsize=12)
    plt.title('Frequências Relativas Observadas vs Esperadas - Dado', fontsize=13, fontweight='bold')
    plt.xticks(x, freq_df['Face'])
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('plots/questao3d_dado_observado_vs_esperado.png', dpi=300)
    plt.close()
    print("Gráfico salvo em: plots/questao3d_dado_observado_vs_esperado.png")
    
    return freq_df, stats_df, lancamentos


def questao_4_futebol():
    """
    Questão 4: Simular 200 partidas com histórico:
    - 43% vitórias
    - 22% empates
    - 35% derrotas
    Calcular frequências esperadas e observadas
    """
    print("\n" + "="*70)
    print("QUESTÃO 4: Simulação de Futebol (200 partidas)")
    print("="*70)
    
    np.random.seed(42)
    # Probabilidades
    probs = [0.43, 0.22, 0.35]  # vitória, empate, derrota
    resultados_possiveis = ['Vitória', 'Empate', 'Derrota']
    
    # Simular 200 partidas
    partidas = np.random.choice(resultados_possiveis, size=200, p=probs)
    
    # Frequências observadas
    valores_unicos, contagens = np.unique(partidas, return_counts=True)
    
    freq_futebol = pd.DataFrame({
        'Resultado': valores_unicos,
        'Frequência_Absoluta': contagens,
        'Frequência_Relativa_Observada': (contagens / len(partidas)).round(4)
    })
    
    # Frequências esperadas
    freq_esperada = pd.DataFrame({
        'Resultado': resultados_possiveis,
        'Probabilidade_Teórica': probs,
        'Frequência_Esperada': np.array(probs) * 200
    })
    
    # Mesclar tabelas
    freq_futebol_completo = freq_futebol.merge(freq_esperada, on='Resultado', how='outer')
    freq_futebol_completo.fillna(0, inplace=True)
    freq_futebol_completo['Frequência_Relativa_Esperada'] = (freq_futebol_completo['Frequência_Esperada'] / 200).round(4)
    freq_futebol_completo['Diferença_Absoluta'] = (freq_futebol_completo['Frequência_Absoluta'] - 
                                                      freq_futebol_completo['Frequência_Esperada']).round(0)
    freq_futebol_completo['Diferença_Relativa'] = (freq_futebol_completo['Frequência_Relativa_Observada'] - 
                                                      freq_futebol_completo['Frequência_Relativa_Esperada']).round(4)
    
    # Ordenar de forma consistente
    ordem = {'Vitória': 0, 'Empate': 1, 'Derrota': 2}
    freq_futebol_completo['ordem'] = freq_futebol_completo['Resultado'].map(ordem)
    freq_futebol_completo = freq_futebol_completo.sort_values('ordem').drop('ordem', axis=1)
    
    print("\nFrequências Observadas vs Esperadas (200 partidas):")
    print(freq_futebol_completo.to_string(index=False))
    
    freq_futebol_completo.to_csv('outputs/questao4_frequencias_futebol.csv', index=False)
    print("\nTabela salva em: outputs/questao4_frequencias_futebol.csv")
    
    # Gráfico comparando observado vs esperado
    plt.figure(figsize=(11, 7))
    x = np.arange(len(freq_futebol_completo))
    width = 0.35
    
    plt.bar(x - width/2, freq_futebol_completo['Frequência_Absoluta'], width, 
            label='Observado', color='#2ecc71', edgecolor='black', alpha=0.8)
    plt.bar(x + width/2, freq_futebol_completo['Frequência_Esperada'], width, 
            label='Esperado', color='#e74c3c', edgecolor='black', alpha=0.8)
    
    plt.xlabel('Resultado da Partida', fontsize=12)
    plt.ylabel('Frequência Absoluta', fontsize=12)
    plt.title('Frequências Observadas vs Esperadas - Futebol (200 partidas)', fontsize=13, fontweight='bold')
    plt.xticks(x, freq_futebol_completo['Resultado'])
    plt.legend(fontsize=11)
    
    # Adicionar valores nas barras
    for i, (obs, esp) in enumerate(zip(freq_futebol_completo['Frequência_Absoluta'], 
                                         freq_futebol_completo['Frequência_Esperada'])):
        plt.text(i - width/2, obs + 2, str(int(obs)), ha='center', fontsize=10)
        plt.text(i + width/2, esp + 2, str(int(esp)), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('plots/questao4_futebol_observado_vs_esperado.png', dpi=300)
    plt.close()
    print("Gráfico salvo em: plots/questao4_futebol_observado_vs_esperado.png")
    
    return freq_futebol_completo


def main():
    print("\n" + "█"*70)
    print("█ " + " "*66 + " █")
    print("█ " + "SIMULAÇÕES DE PROBABILIDADE - ANÁLISE ESTATÍSTICA".center(66) + " █")
    print("█ " + " "*66 + " █")
    print("█"*70)
    
    # Questão 1
    lancamentos, recodificado = questao_1_moeda()
    
    # Questão 2
    freq_moeda = questao_2_moeda(lancamentos, recodificado)
    
    # Questão 3
    freq_dado, stats_dado, lancamentos_dado = questao_3_dado()
    
    # Questão 4
    freq_futebol = questao_4_futebol()
    
    print("\n" + "="*70)
    print("RESUMO FINAL")
    print("="*70)
    print("\nArquivos gerados em 'outputs/':")
    print("  ✓ questao1_sequencia_moeda.csv")
    print("  ✓ questao2_frequencias_moeda.csv")
    print("  ✓ questao3a_frequencias_dado.csv")
    print("  ✓ questao3b_estatisticas_dado.csv")
    print("  ✓ questao4_frequencias_futebol.csv")
    print("\nGráficos gerados em 'plots/':")
    print("  ✓ questao3c_frequencias_dado_barras.png")
    print("  ✓ questao3d_dado_observado_vs_esperado.png")
    print("  ✓ questao4_futebol_observado_vs_esperado.png")
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
