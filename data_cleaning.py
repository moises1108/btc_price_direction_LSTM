import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import gym
from gym import spaces
from stable_baselines3 import PPO



def create_sequences_price_prediction(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length, 0])  # Predicting the 'close' price
    return np.array(sequences), np.array(targets)
def create_sequences_price_direction(data,seq_length, target):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(target[i + seq_length])
    return np.array(sequences), np.array(targets)

def get_crypto_news():
    crypto_news = pd.read_csv('./crypto_news/cryptonews.csv')
    print(crypto_news.columns)
    print(crypto_news.head)



# predicts the price of the next day
def get_crypto_prices(cryptos):
    if type(cryptos) != list:
        print("Did not provide a list when trying to get prices")
        return TypeError
    
    # 1. Create initial df where we append each CSV
    columns = {"ticker":[], "date":[], "open":[], "high":[], "low":[], "close":[]}
    crypto_main_df = pd.DataFrame(data=columns)

    for crypto in cryptos:
            try:
                crypto_price_df = pd.read_csv(f'./crypto_prices/{crypto}.csv')
                crypto_main_df = pd.concat([crypto_main_df, crypto_price_df], ignore_index=True)
            except FileNotFoundError as e:
                print(f"Could not find file {crypto}")
    # Adds derived fields MA_10, MA_50, RSI, MACD, MACD_signal, MACD_diff
    crypto_main_df['MA_10'] = crypto_main_df['close'].rolling(window=10).mean()
    crypto_main_df['MA_50'] = crypto_main_df['close'].rolling(window=50).mean()
    crypto_main_df['RSI'] = ta.momentum.RSIIndicator(crypto_main_df['close'], window=14).rsi()
    crypto_main_df['MACD'] = ta.trend.MACD(crypto_main_df['close']).macd()
    crypto_main_df['MACD_signal'] = ta.trend.MACD(crypto_main_df['close']).macd_signal()
    crypto_main_df['MACD_diff'] = ta.trend.MACD(crypto_main_df['close']).macd_diff()

    # 2. Handle missing values - specifically MA
    crypto_main_df.fillna(method="bfill", inplace=True)
    crypto_main_df.fillna(method="ffill", inplace=True)

    # 3. Normalise data
    # Select features for normalization
    features = ['close', 'MA_10', 'MA_50', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff']

    # Normalize the features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(crypto_main_df[features])

    # Convert to DataFrame
    scaled_df = pd.DataFrame(scaled_features, columns=features)
    print(scaled_df)


    #4. Create sequences
    sequence_length = 50
    X, y = create_sequences_price_prediction(scaled_features, sequence_length)

    #5. Split Data into Training and Testing Sets
    # Split the data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    #6. Build the Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    #7. Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

    #8. Evaluate the model
    # Making predictions
    # Make predictions
    predicted_prices = model.predict(X_test)

    # Inverse transform the predictions and actual values
    predicted_prices = scaler.inverse_transform(np.concatenate((predicted_prices, np.zeros((predicted_prices.shape[0], scaled_df.shape[1] - 1))), axis=1))[:, 0]
    actual_prices = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_df.shape[1] - 1))), axis=1))[:, 0]
    
    # Plot results
    plt.figure(figsize=(14, 5))
    plt.plot(actual_prices, color='blue', label='Actual Prices')
    plt.plot(predicted_prices, color='red', label='Predicted Prices')
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Returns a df with all the requested crypto prices
    return crypto_main_df 

# predicts if it will go up or down on next
def predict_side(cryptos):
    if type(cryptos) != list:
        print("Did not provide a list when trying to get prices")
        return TypeError
    
    # 1. Create initial df where we append each CSV
    columns = {"ticker":[], "date":[], "open":[], "high":[], "low":[], "close":[]}
    crypto_main_df = pd.DataFrame(data=columns)

    for crypto in cryptos:
            try:
                crypto_price_df = pd.read_csv(f'./crypto_prices/{crypto}.csv')
                crypto_main_df = pd.concat([crypto_main_df, crypto_price_df], ignore_index=True)
            except FileNotFoundError as e:
                print(f"Could not find file {crypto}")
    # Adds derived fields MA_10, MA_50, RSI, MACD, MACD_signal, MACD_diff
    crypto_main_df['MA_10'] = crypto_main_df['close'].rolling(window=10).mean()
    crypto_main_df['MA_50'] = crypto_main_df['close'].rolling(window=50).mean()
    crypto_main_df['RSI'] = ta.momentum.RSIIndicator(crypto_main_df['close'], window=14).rsi()
    crypto_main_df['MACD'] = ta.trend.MACD(crypto_main_df['close']).macd()
    crypto_main_df['MACD_signal'] = ta.trend.MACD(crypto_main_df['close']).macd_signal()
    crypto_main_df['MACD_diff'] = ta.trend.MACD(crypto_main_df['close']).macd_diff()
    
    # 2. Handle missing values - specifically MA
    crypto_main_df.fillna(method="bfill", inplace=True)
    crypto_main_df.fillna(method="ffill", inplace=True)
    
    # Add the direction of price movement
    crypto_main_df['Price_Up'] = (crypto_main_df['close'].shift(-1) > crypto_main_df['close']).astype(int)
    crypto_main_df.dropna(inplace=True)

    # 3. Normalise data
    # Select features for normalization
    features = ['close', 'MA_10', 'MA_50', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff']

    # Normalize the features
    scaler = MinMaxScaler()

    #4. Create sequences
    sequence_length = 50
    features = crypto_main_df[['close', 'MA_10', 'MA_50', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff']]
    target = crypto_main_df['Price_Up']
    scaled_features = scaler.fit_transform(features)
    X, y = create_sequences_price_direction(scaled_features, sequence_length, target)


    #5. Split Data into Training and Testing Sets
    # Split the data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    #6. Build the Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #7. Train the model
    history = model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.2)

    #8. Evaluate the model
    # Making predictions
    # Make predictions
    probabilities = model.predict(X_test)
    predictions = (probabilities > 0.5).astype(int)

    # Evaluate the model's performance using accuracy, precision, recall, and other relevant metrics.
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy}')
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

# reinforcement learning
class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.current_step = 0
        self.balance = 1000
        self.net_worth = 1000
        self.shares_held = 0
        self.trade_history = []
        
        # Actions: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(df.shape[1],), dtype=np.float32)
        
    def reset(self):
        self.current_step = 0
        self.balance = 1000
        self.net_worth = 1000
        self.shares_held = 0
        self.trade_history = []
        return self.df.iloc[self.current_step].values
    
    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        self.current_step += 1
        
        if action == 1:  # Buy
            self.shares_held += self.balance // current_price
            self.balance -= self.shares_held * current_price
        elif action == 2:  # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0
        
        self.net_worth = self.balance + self.shares_held * current_price
        
        reward = self.net_worth - 1000  # Reward is change in net worth
        done = self.current_step >= len(self.df) - 1
        obs = self.df.iloc[self.current_step].values
        
        self.trade_history.append({
            'step': self.current_step,
            'price': current_price,
            'action': action,
            'balance': self.balance,
            'net_worth': self.net_worth,
            'shares_held': self.shares_held
        })
        
        return obs, reward, done, {}
    
    def render(self, mode='human'):
        pass


def reinforcement_learning(cryptos):
    if type(cryptos) != list:
        print("Did not provide a list when trying to get prices")
        return TypeError
    
    # 1. Create initial df where we append each CSV
    columns = {"ticker":[], "date":[], "open":[], "high":[], "low":[], "close":[]}
    crypto_main_df = pd.DataFrame(data=columns)

    for crypto in cryptos:
            try:
                crypto_price_df = pd.read_csv(f'./crypto_prices/{crypto}.csv')
                crypto_main_df = pd.concat([crypto_main_df, crypto_price_df], ignore_index=True)
            except FileNotFoundError as e:
                print(f"Could not find file {crypto}")
    # Ensure the date column is in datetime format
    crypto_main_df['date'] = pd.to_datetime(crypto_main_df['date'])

    # ONLY WORKS WITH ONE CRYPTO
    # Sort by date
    crypto_main_df.sort_values(by='date', inplace=True)
    
    # Create technical indicators
    crypto_main_df['MA_10'] = crypto_main_df['close'].rolling(window=10).mean()
    crypto_main_df['MA_50'] = crypto_main_df['close'].rolling(window=50).mean()
    crypto_main_df['RSI'] = ta.momentum.RSIIndicator(crypto_main_df['close'], window=14).rsi()
    crypto_main_df['MACD'] = ta.trend.MACD(crypto_main_df['close']).macd()
    crypto_main_df['MACD_signal'] = ta.trend.MACD(crypto_main_df['close']).macd_signal()
    crypto_main_df['MACD_diff'] = ta.trend.MACD(crypto_main_df['close']).macd_diff()

    # Fill missing values
    crypto_main_df.fillna(method='bfill', inplace=True)
    crypto_main_df.fillna(method='ffill', inplace=True)

    # Normalize the data
    features = ['close', 'MA_10', 'MA_50', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(crypto_main_df[features])
    crypto_main_df[features] = scaled_features

    # Select and train an RL Agent
    env = TradingEnv(crypto_main_df)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)

    #Back testing and evaluation
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

    # Plot results
    trade_history = pd.DataFrame(env.trade_history)

    plt.figure(figsize=(14, 7))
    plt.plot(crypto_main_df['date'], crypto_main_df['close'], label='Close Price')
    buy_signals = trade_history[trade_history['action'] == 1]
    sell_signals = trade_history[trade_history['action'] == 2]
    plt.scatter(buy_signals['step'], buy_signals['price'], marker='^', color='g', label='Buy', alpha=1)
    plt.scatter(sell_signals['step'], sell_signals['price'], marker='v', color='r', label='Sell', alpha=1)
    plt.title('Trading Bot Performance')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Print profit or loss for each trade
    trade_history['profit'] = trade_history['net_worth'].diff().fillna(0)
    print(trade_history[['step', 'price', 'action', 'profit']])

    # Print total profit or loss
    total_profit = trade_history['profit'].sum()
    print(f'Total Profit: {total_profit}')    


reinforcement_learning(["BTC"])