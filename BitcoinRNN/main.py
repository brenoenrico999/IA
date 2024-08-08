import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
import matplotlib.pyplot as plt

# Carregar os dados históricos do Bitcoin
# Substitua esta parte pelo código de carregamento dos dados da Binance API ou outra fonte
# Exemplo de carregamento de dados de um arquivo CSV:
bitcoin_data = pd.read_csv('bitcoin_price_data.csv')

# Pré-processamento dos dados
scaler = MinMaxScaler()
bitcoin_data['Close'] = scaler.fit_transform(bitcoin_data['Close'].values.reshape(-1, 1))

# Definir as sequências temporais para previsão
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

# Hiperparâmetros
sequence_length = 24  # Utilizar dados de um dia (24 horas) para prever a próxima hora
train_size = int(len(bitcoin_data) * 0.8)  # 80% dos dados para treinamento

# Criar sequências temporais
data = bitcoin_data['Close'].values
train_data = data[:train_size]
test_data = data[train_size:]
X_train = create_sequences(train_data, sequence_length)
y_train = data[sequence_length:train_size + sequence_length]
X_test = create_sequences(test_data, sequence_length)
y_test = data[sequence_length + train_size:]

# Adicionar dimensão de características para LSTM/GRU
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Construir o modelo RNN com LSTM ou GRU
model = Sequential()
# Descomente a configuração desejada:

# Usar LSTM
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))

# Usar GRU
# model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
# model.add(GRU(units=50))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Avaliar o modelo
loss = model.evaluate(X_test, y_test)
print(f"Loss (Erro Médio Quadrático) no conjunto de teste: {loss}")

# Fazer previsões
predicted_values = model.predict(X_test)

# Reverter a normalização para obter os valores reais
predicted_values = scaler.inverse_transform(predicted_values)
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

# Prever o valor da próxima hora
last_sequence = data[-sequence_length:]
last_sequence = last_sequence.reshape((1, sequence_length, 1))  # Ajustar formato para o modelo
next_hour_prediction = model.predict(last_sequence)
next_hour_prediction = scaler.inverse_transform(next_hour_prediction)
print(f"Valor previsto para a próxima hora: {next_hour_prediction[0][0]}")

# Visualizar os resultados
plt.figure(figsize=(14, 7))
plt.plot(range(len(data)), scaler.inverse_transform(data.reshape(-1, 1)), label='Dados Reais')
plt.plot(range(len(train_data) + sequence_length, len(data)), predicted_values, label='Previsões')
plt.legend()
plt.xlabel('Tempo')
plt.ylabel('Preço do Bitcoin')
plt.title('Previsão do Preço do Bitcoin')
plt.show()
