# Previsão do Preço do Bitcoin com RNNs, LSTM e GRU

Este projeto utiliza redes neurais recorrentes (RNNs), especificamente LSTM (Long Short-Term Memory) ou GRU (Gated Recurrent Unit), para prever o preço do Bitcoin com base em dados históricos.

## Introdução
Redes Neurais Recorrentes (RNNs) são um tipo de rede neural que é capaz de processar sequências de dados e é frequentemente usada para análise de séries temporais. Em comparação com redes neurais feedforward, RNNs possuem conexões que formam ciclos, permitindo que a informação persista e seja usada para prever futuras entradas.

## LSTM vs. GRU

LSTM (Long Short-Term Memory) e GRU (Gated Recurrent Unit) são variações de RNNs que foram projetadas para lidar com problemas de longa dependência temporal e superarem algumas limitações das RNNs básicas.

### LSTM (Long Short-Term Memory)

#### Vantagens:

Gerenciamento de Dependências Longas: LSTM pode capturar dependências de longo prazo devido à sua estrutura de células de memória.

Desempenho Consistente: Geralmente oferece bom desempenho em tarefas de previsão de séries temporais complexas e prolongadas.

#### Desvantagens:

Complexidade: A estrutura interna é mais complexa do que a de RNNs simples e GRUs, o que pode levar a um tempo de treinamento mais longo.

Demais Parâmetros: Maior número de parâmetros pode aumentar o risco de sobreajuste, especialmente com dados limitados.

### GRU (Gated Recurrent Unit)

#### Vantagens:

Simplicidade: GRUs têm uma estrutura mais simples em comparação com LSTMs, o que pode levar a um treinamento mais rápido.

Menos Parâmetros: Menos parâmetros do que o LSTM, o que pode reduzir o risco de sobreajuste e acelerar o treinamento.

#### Desvantagens:

Desempenho Variável: Pode não capturar dependências de longo prazo tão bem quanto LSTMs em algumas situações, embora isso dependa do problema específico e dos dados.

## Código

O código neste repositório utiliza tanto LSTM quanto GRU para prever o preço do Bitcoin com base em dados históricos. Você pode escolher qual modelo usar comentando ou descomentando as linhas correspondentes no código.

### Estrutura do Código

Carregar e Pré-processar Dados: Os dados históricos do Bitcoin são carregados e normalizados.

Criar Sequências Temporais: Sequências de dados são criadas para treinamento e teste do modelo.

Construir o Modelo: O modelo pode ser construído com LSTM ou GRU.

Treinar o Modelo: O modelo é treinado usando os dados históricos.

Avaliar e Prever: O modelo é avaliado e utilizado para prever o valor da próxima hora.

Visualizar Resultados: Gráficos são gerados para comparar previsões com dados reais.
Executando o Código

### Instale as dependências necessárias:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
```

Substitua o caminho do arquivo CSV no código para o local onde seus dados estão armazenados.


## Observações

**Certifique-se de ajustar os hiperparâmetros e o pré-processamento dos dados conforme necessário para otimizar o desempenho do modelo.**

A previsão de preços de ativos financeiros pode ser influenciada por vários fatores externos. Considere isso ao interpretar os resultados.

## Licença
Este projeto está licenciado sob a MIT License.