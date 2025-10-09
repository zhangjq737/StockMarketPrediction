#!/usr/bin/env python
# coding: utf-8

# In[33]:


# Cell 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully!")


# In[48]:


# Cell 2: Download Stock Data
ticker = 'AAPL'
start_date = '2019-01-01'
end_date = '2024-01-01'

df = yf.download(ticker, start=start_date, end=end_date)
print(f"\nData shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")

plt.figure(figsize=(14, 5))
plt.plot(df['Close'])
plt.title(f'{ticker} Stock Price History')
plt.xlabel('Date')
plt.ylabel('Close Price ($)')
plt.grid(True)
plt.savefig('stock_price_history.png', dpi=300, bbox_inches='tight')
plt.show()


# In[49]:


# Cell 3: Data Preprocessing
data = df['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

print(f"\nTraining data size: {len(train_data)}")
print(f"Testing data size: {len(test_data)}")


# In[50]:


# Cell 4: Create Sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 60
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"\nX_train shape: {X_train_lstm.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test_lstm.shape}")
print(f"y_test shape: {y_test.shape}")


# In[51]:


# Cell 5: LSTM Model
print("=" * 50)
print("MODEL 1: LSTM")
print("=" * 50)

lstm_model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

print("\nTraining LSTM model...")
history_lstm = lstm_model.fit(
    X_train_lstm, y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.1,
    verbose=1
)

lstm_predictions = lstm_model.predict(X_test_lstm)
lstm_predictions = scaler.inverse_transform(lstm_predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

lstm_rmse = np.sqrt(mean_squared_error(y_test_actual, lstm_predictions))
lstm_mae = mean_absolute_error(y_test_actual, lstm_predictions)
lstm_r2 = r2_score(y_test_actual, lstm_predictions)

print(f"\nLSTM Performance:")
print(f"RMSE: ${lstm_rmse:.2f}")
print(f"MAE: ${lstm_mae:.2f}")
print(f"R² Score: {lstm_r2:.4f}")


# In[52]:


# Cell 6: GRU Model
print("\n" + "=" * 50)
print("MODEL 2: GRU")
print("=" * 50)

gru_model = Sequential([
    GRU(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
    Dropout(0.2),
    GRU(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

gru_model.compile(optimizer='adam', loss='mean_squared_error')

print("\nTraining GRU model...")
history_gru = gru_model.fit(
    X_train_lstm, y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.1,
    verbose=1
)

gru_predictions = gru_model.predict(X_test_lstm)
gru_predictions = scaler.inverse_transform(gru_predictions)

gru_rmse = np.sqrt(mean_squared_error(y_test_actual, gru_predictions))
gru_mae = mean_absolute_error(y_test_actual, gru_predictions)
gru_r2 = r2_score(y_test_actual, gru_predictions)

print(f"\nGRU Performance:")
print(f"RMSE: ${gru_rmse:.2f}")
print(f"MAE: ${gru_mae:.2f}")
print(f"R² Score: {gru_r2:.4f}")


# In[53]:


# Cell 7: Dense Neural Network
print("\n" + "=" * 50)
print("MODEL 3: Dense Neural Network")
print("=" * 50)

dense_model = Sequential([
    Dense(units=128, activation='relu', input_shape=(seq_length,)),
    Dropout(0.2),
    Dense(units=64, activation='relu'),
    Dropout(0.2),
    Dense(units=32, activation='relu'),
    Dense(units=1)
])

dense_model.compile(optimizer='adam', loss='mean_squared_error')

print("\nTraining Dense NN model...")
history_dense = dense_model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.1,
    verbose=1
)

dense_predictions = dense_model.predict(X_test)
dense_predictions = scaler.inverse_transform(dense_predictions)

dense_rmse = np.sqrt(mean_squared_error(y_test_actual, dense_predictions))
dense_mae = mean_absolute_error(y_test_actual, dense_predictions)
dense_r2 = r2_score(y_test_actual, dense_predictions)

print(f"\nDense NN Performance:")
print(f"RMSE: ${dense_rmse:.2f}")
print(f"MAE: ${dense_mae:.2f}")
print(f"R² Score: {dense_r2:.4f}")


# In[54]:


# Cell 8: Model Comparison
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

comparison_df = pd.DataFrame({
    'Model': ['LSTM', 'GRU', 'Dense NN'],
    'RMSE': [lstm_rmse, gru_rmse, dense_rmse],
    'MAE': [lstm_mae, gru_mae, dense_mae],
    'R² Score': [lstm_r2, gru_r2, dense_r2]
})

print(comparison_df.to_string(index=False))

best_model_idx = comparison_df['RMSE'].idxmin()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
print(f"\nBest Model: {best_model_name} (Lowest RMSE)")


# In[55]:


# Cell 9: Visualize Predictions
plt.figure(figsize=(16, 10))

plt.subplot(3, 1, 1)
plt.plot(y_test_actual, label='Actual Price', color='blue', linewidth=2)
plt.plot(lstm_predictions, label='LSTM Predictions', color='red', linewidth=2)
plt.title(f'LSTM Model - RMSE: ${lstm_rmse:.2f}', fontsize=12, fontweight='bold')
plt.xlabel('Time')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(y_test_actual, label='Actual Price', color='blue', linewidth=2)
plt.plot(gru_predictions, label='GRU Predictions', color='green', linewidth=2)
plt.title(f'GRU Model - RMSE: ${gru_rmse:.2f}', fontsize=12, fontweight='bold')
plt.xlabel('Time')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(y_test_actual, label='Actual Price', color='blue', linewidth=2)
plt.plot(dense_predictions, label='Dense NN Predictions', color='orange', linewidth=2)
plt.title(f'Dense NN Model - RMSE: ${dense_rmse:.2f}', fontsize=12, fontweight='bold')
plt.xlabel('Time')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_predictions.png', dpi=300, bbox_inches='tight')
plt.show()


# In[56]:


# Cell 10: Training History
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history_lstm.history['loss'], label='Training Loss')
plt.plot(history_lstm.history['val_loss'], label='Validation Loss')
plt.title('LSTM Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(history_gru.history['loss'], label='Training Loss')
plt.plot(history_gru.history['val_loss'], label='Validation Loss')
plt.title('GRU Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(history_dense.history['loss'], label='Training Loss')
plt.plot(history_dense.history['val_loss'], label='Validation Loss')
plt.title('Dense NN Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()


# In[57]:


# Cell 11: Save Best Model
if best_model_name == 'LSTM':
    lstm_model.save('best_model.h5')
elif best_model_name == 'GRU':
    gru_model.save('best_model.h5')
else:
    dense_model.save('best_model.h5')

print(f"\nBest model ({best_model_name}) saved as 'best_model.h5'")
print("\nProject completed successfully!")


# In[66]:


# Cell 12: GRU From Scratch Implementation

import numpy as np

class GRUFromScratch:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights with Xavier initialization
        limit = np.sqrt(6 / (input_size + hidden_size))

        # Update gate weights
        self.Wz = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.Uz = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.bz = np.zeros((hidden_size, 1))

        # Reset gate weights
        self.Wr = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.Ur = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.br = np.zeros((hidden_size, 1))

        # Candidate hidden state weights
        self.Wh = np.random.uniform(-limit, limit, (hidden_size, input_size))
        self.Uh = np.random.uniform(-limit, limit, (hidden_size, hidden_size))
        self.bh = np.zeros((hidden_size, 1))

        # Output layer weights
        self.Wy = np.random.uniform(-limit, limit, (output_size, hidden_size))
        self.by = np.zeros((output_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))

    def forward_pass(self, X):
        """
        Forward pass through GRU
        X: input sequence (seq_length, input_size)
        """
        seq_length = X.shape[0]
        h = np.zeros((self.hidden_size, 1))

        self.inputs = []
        self.hidden_states = [h.copy()]
        self.z_gates = []
        self.r_gates = []
        self.h_candidates = []

        for t in range(seq_length):
            x_t = X[t].reshape(-1, 1)
            self.inputs.append(x_t)

            # Update gate
            z_t = self.sigmoid(np.dot(self.Wz, x_t) + np.dot(self.Uz, h) + self.bz)
            self.z_gates.append(z_t)

            # Reset gate
            r_t = self.sigmoid(np.dot(self.Wr, x_t) + np.dot(self.Ur, h) + self.br)
            self.r_gates.append(r_t)

            # Candidate hidden state
            h_candidate = self.tanh(np.dot(self.Wh, x_t) + np.dot(self.Uh, r_t * h) + self.bh)
            self.h_candidates.append(h_candidate)

            # New hidden state
            h = z_t * h + (1 - z_t) * h_candidate
            self.hidden_states.append(h.copy())

        # Output
        y = np.dot(self.Wy, h) + self.by
        return y, h

    def backward_pass(self, X, y_true, y_pred, h_final):
        """
        Backward pass through GRU using BPTT
        """
        seq_length = len(X)

        # Initialize gradients
        dWz = np.zeros_like(self.Wz)
        dUz = np.zeros_like(self.Uz)
        dbz = np.zeros_like(self.bz)

        dWr = np.zeros_like(self.Wr)
        dUr = np.zeros_like(self.Ur)
        dbr = np.zeros_like(self.br)

        dWh = np.zeros_like(self.Wh)
        dUh = np.zeros_like(self.Uh)
        dbh = np.zeros_like(self.bh)

        # Output layer gradient
        dy = 2 * (y_pred - y_true) / y_true.size
        dWy = np.dot(dy, h_final.T)
        dby = dy

        # Backpropagate through time
        dh_next = np.dot(self.Wy.T, dy)

        for t in reversed(range(seq_length)):
            x_t = self.inputs[t]
            h_prev = self.hidden_states[t]
            h_curr = self.hidden_states[t + 1]
            z_t = self.z_gates[t]
            r_t = self.r_gates[t]
            h_candidate = self.h_candidates[t]

            # Gradient of hidden state
            dh = dh_next

            # Gradient through update gate
            dh_candidate = dh * (1 - z_t)
            dz = dh * (h_prev - h_candidate)

            # Gradient through candidate hidden state
            dh_candidate_raw = dh_candidate * (1 - h_candidate ** 2)
            dWh += np.dot(dh_candidate_raw, x_t.T)
            dUh += np.dot(dh_candidate_raw, (r_t * h_prev).T)
            dbh += dh_candidate_raw

            # Gradient through reset gate
            dr = np.dot(self.Uh.T, dh_candidate_raw) * h_prev
            dr_raw = dr * r_t * (1 - r_t)
            dWr += np.dot(dr_raw, x_t.T)
            dUr += np.dot(dr_raw, h_prev.T)
            dbr += dr_raw

            # Gradient through update gate
            dz_raw = dz * z_t * (1 - z_t)
            dWz += np.dot(dz_raw, x_t.T)
            dUz += np.dot(dz_raw, h_prev.T)
            dbz += dz_raw

            # Gradient to previous hidden state
            dh_next = (np.dot(self.Uz.T, dz_raw) + 
                      np.dot(self.Ur.T, dr_raw) + 
                      np.dot(self.Uh.T, dh_candidate_raw) * r_t + 
                      dh * z_t)

        # Clip gradients to prevent exploding gradients
        max_grad = 5
        for grad in [dWz, dUz, dbz, dWr, dUr, dbr, dWh, dUh, dbh, dWy, dby]:
            np.clip(grad, -max_grad, max_grad, out=grad)

        return dWz, dUz, dbz, dWr, dUr, dbr, dWh, dUh, dbh, dWy, dby

    def update_weights(self, dWz, dUz, dbz, dWr, dUr, dbr, dWh, dUh, dbh, dWy, dby):
        """Update weights using gradient descent"""
        self.Wz -= self.learning_rate * dWz
        self.Uz -= self.learning_rate * dUz
        self.bz -= self.learning_rate * dbz

        self.Wr -= self.learning_rate * dWr
        self.Ur -= self.learning_rate * dUr
        self.br -= self.learning_rate * dbr

        self.Wh -= self.learning_rate * dWh
        self.Uh -= self.learning_rate * dUh
        self.bh -= self.learning_rate * dbh

        self.Wy -= self.learning_rate * dWy
        self.by -= self.learning_rate * dby

    def train(self, X_train, y_train, epochs=100, batch_size=32):
        """Train the GRU model"""
        n_samples = len(X_train)
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0

            # Shuffle data
            indices = np.random.permutation(n_samples)

            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:min(i + batch_size, n_samples)]
                batch_loss = 0

                # Accumulate gradients over batch
                dWz_batch = np.zeros_like(self.Wz)
                dUz_batch = np.zeros_like(self.Uz)
                dbz_batch = np.zeros_like(self.bz)
                dWr_batch = np.zeros_like(self.Wr)
                dUr_batch = np.zeros_like(self.Ur)
                dbr_batch = np.zeros_like(self.br)
                dWh_batch = np.zeros_like(self.Wh)
                dUh_batch = np.zeros_like(self.Uh)
                dbh_batch = np.zeros_like(self.bh)
                dWy_batch = np.zeros_like(self.Wy)
                dby_batch = np.zeros_like(self.by)

                for idx in batch_indices:
                    X_seq = X_train[idx]
                    y_true = np.array([[y_train[idx]]])

                    # Forward pass
                    y_pred, h_final = self.forward_pass(X_seq)

                    # Calculate loss (MSE)
                    loss = np.mean((y_pred - y_true) ** 2)
                    batch_loss += loss

                    # Backward pass
                    grads = self.backward_pass(X_seq, y_true, y_pred, h_final)
                    dWz, dUz, dbz, dWr, dUr, dbr, dWh, dUh, dbh, dWy, dby = grads

                    # Accumulate gradients
                    dWz_batch += dWz
                    dUz_batch += dUz
                    dbz_batch += dbz
                    dWr_batch += dWr
                    dUr_batch += dUr
                    dbr_batch += dbr
                    dWh_batch += dWh
                    dUh_batch += dUh
                    dbh_batch += dbh
                    dWy_batch += dWy
                    dby_batch += dby

                # Average gradients over batch
                batch_size_actual = len(batch_indices)
                dWz_batch /= batch_size_actual
                dUz_batch /= batch_size_actual
                dbz_batch /= batch_size_actual
                dWr_batch /= batch_size_actual
                dUr_batch /= batch_size_actual
                dbr_batch /= batch_size_actual
                dWh_batch /= batch_size_actual
                dUh_batch /= batch_size_actual
                dbh_batch /= batch_size_actual
                dWy_batch /= batch_size_actual
                dby_batch /= batch_size_actual

                # Update weights
                self.update_weights(dWz_batch, dUz_batch, dbz_batch, 
                                   dWr_batch, dUr_batch, dbr_batch,
                                   dWh_batch, dUh_batch, dbh_batch,
                                   dWy_batch, dby_batch)

                epoch_loss += batch_loss / batch_size_actual

            avg_loss = epoch_loss / (n_samples / batch_size)
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        return losses

    def predict(self, X):
        """Make predictions"""
        predictions = []
        for x_seq in X:
            y_pred, _ = self.forward_pass(x_seq)
            predictions.append(y_pred[0, 0])
        return np.array(predictions)


# In[67]:


# Cell 13: Train GRU From Scratch
print("=" * 50)
print("GRU MODEL FROM SCRATCH")
print("=" * 50)

# Initialize model
gru_scratch = GRUFromScratch(
    input_size=1,
    hidden_size=50,
    output_size=1,
    learning_rate=0.001
)

# Prepare data for from-scratch model
X_train_scratch = X_train_lstm.reshape(X_train_lstm.shape[0], X_train_lstm.shape[1], 1)
X_test_scratch = X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[1], 1)

print(f"\nTraining GRU from scratch...")
print(f"Training samples: {len(X_train_scratch)}")
print(f"Sequence length: {X_train_scratch.shape[1]}")

# Train the model
losses = gru_scratch.train(X_train_scratch, y_train, epochs=100, batch_size=32)

# Make predictions
print("\nMaking predictions...")
scratch_predictions_scaled = gru_scratch.predict(X_test_scratch)
scratch_predictions = scaler.inverse_transform(scratch_predictions_scaled.reshape(-1, 1))

# Calculate metrics
scratch_rmse = np.sqrt(mean_squared_error(y_test_actual, scratch_predictions))
scratch_mae = mean_absolute_error(y_test_actual, scratch_predictions)
scratch_r2 = r2_score(y_test_actual, scratch_predictions)

print(f"\nGRU From Scratch Performance:")
print(f"RMSE: ${scratch_rmse:.2f}")
print(f"MAE: ${scratch_mae:.2f}")
print(f"R² Score: {scratch_r2:.4f}")


# In[69]:


# Cell 14: Compare Library vs From Scratch
print("\n" + "=" * 70)
print("LIBRARY GRU vs FROM SCRATCH GRU COMPARISON")
print("=" * 70)

comparison_scratch_df = pd.DataFrame({
    'Model': ['GRU (Keras)', 'GRU (From Scratch)'],
    'RMSE': [gru_rmse, scratch_rmse],
    'MAE': [gru_mae, scratch_mae],
    'R² Score': [gru_r2, scratch_r2]
})

print(comparison_scratch_df.to_string(index=False))


# In[70]:


# Cell 15: Visualize From Scratch Results
plt.figure(figsize=(16, 8))

# Plot 1: Predictions comparison
plt.subplot(2, 1, 1)
plt.plot(y_test_actual, label='Actual Price', color='blue', linewidth=2)
plt.plot(gru_predictions, label='GRU (Keras)', color='green', linewidth=2, alpha=0.7)
plt.plot(scratch_predictions, label='GRU (From Scratch)', color='red', linewidth=2, alpha=0.7)
plt.title('GRU Comparison: Keras vs From Scratch', fontsize=14, fontweight='bold')
plt.xlabel('Time')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Training loss
plt.subplot(2, 1, 2)
plt.plot(losses, label='Training Loss', color='purple', linewidth=2)
plt.title('GRU From Scratch - Training Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gru_from_scratch_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFrom scratch implementation completed!")
print("Visualization saved as 'gru_from_scratch_comparison.png'")


# In[ ]:




