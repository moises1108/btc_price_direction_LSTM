# Predicting BTC's price direction

### Libraries
- ta: for technical analysis
- pandas
- sklearn
- seaborn
- matplotlib
- numpy
- tensorflow

### Data source:
- Crypto news data [source](https://www.kaggle.com/datasets/oliviervha/crypto-news)
- Crypto price data [source](https://www.kaggle.com/datasets/svaningelgem/crypto-currencies-daily-prices?select=BTC.csv)

### Models
1. LSTM
2. Reinforcement learning
3. Both

## Non-technical write-up
An LSTM (Long Short-Term Memory) model is a type of neural network designed to recognise patterns in sequences of data, such as time series data. For predicting the direction of Bitcoin prices, the model learns from historical price patterns, various technical indicators and news sentiment data.

The LSTM model was trained to classify whether the price would go up or down the next day. The classification report shows the model's performance:

The model turns out to be highly accurate (93%) at predicting when prices don't increase (precision of 0.93) but struggles to predict price increases (precision of 0.55 and recall of 0.10), indicating it performs better in stable or declining markets.

## Data source
- Crypto news data [source](https://www.kaggle.com/datasets/oliviervha/crypto-news)
- Crypto price data [source](https://www.kaggle.com/datasets/svaningelgem/crypto-currencies-daily-prices?select=BTC.csv)

## Models 
LSTM, regression and random forests were used but LSTM was the final choice for the following reasons.

1. Sequential Data Handling
Bitcoin price data is inherently sequential, as prices are recorded over time. LSTMs are designed to capture temporal dependencies and patterns in time series data, making them well-suited for tasks involving historical price movements.

2. Memory Capacity
LSTMs have a unique architecture that allows them to remember information over long periods. This memory capability is crucial for financial data, where past prices and trends can significantly influence future movements. The model can learn and remember patterns that span days, weeks, or even longer.

3. Non-Linear Relationships
The relationship between various factors (like price, volume, and technical indicators) and future price movements is often non-linear. LSTMs, with their complex structures, can capture these non-linear relationships more effectively than traditional linear models.

4. Handling Noise
Financial markets are noisy, and traditional methods may struggle to differentiate between signal and noise. LSTMs are robust to such fluctuations, as their memory cells can filter out irrelevant information and focus on relevant patterns.

5. Integration of Multiple Features
LSTMs can seamlessly integrate multiple input features, such as historical prices, technical indicators, and sentiment analysis from news data. This capability enables the model to leverage a rich set of information, improving its predictive power.

6. Flexibility with Input Length
Unlike many traditional models, LSTMs can handle varying input lengths, making them flexible for time series data with different time horizons or sampling rates.

## HYPERPARAMETER OPTIMSATION
1. Number of LSTM Units: This parameter defines the number of memory units in each LSTM layer. More units can allow the model to learn more complex patterns but can also lead to overfitting.

Optimization: Experiment with different numbers of units (e.g., 50, 100, 200) and use cross-validation to assess performance.

2. Number of Layers: You can stack multiple LSTM layers. Adding layers can help capture more complex features, but it also increases the risk of overfitting.

Optimization: Start with a simple model and gradually add layers to see if performance improves.

3. Dropout Rate: Dropout is a regularization technique used to prevent overfitting by randomly dropping a fraction of neurons during training. The dropout rate determines the proportion of neurons to drop.

Optimization: Experiment with different dropout rates (e.g., 0.1, 0.2, 0.5) and monitor the validation loss to find an optimal value.

4. Learning Rate: This parameter controls the step size during optimization. A smaller learning rate may take longer to converge but can lead to better performance.

Optimization: Use techniques like learning rate scheduling or a learning rate finder to identify an optimal learning rate.

5. Batch Size: The number of samples processed before the model is updated. Smaller batch sizes often lead to better generalization but increase training time.

Optimization: Test different batch sizes (e.g., 16, 32, 64) and evaluate their impact on training stability and performance.

6. Epochs: The number of complete passes through the training dataset. Too few epochs may lead to underfitting, while too many can cause overfitting.

Optimization: Monitor training and validation loss to determine the best number of epochs, potentially using early stopping to prevent overfitting.

## Results
Performance Metrics
- Accuracy: 93%
- Loss: 0.20 (train), 0.21 (validation)
- Precision:
    - Class 0 (price down): 0.93
    - Class 1 (price up): 0.55
- Recall:
    - Class 0: 0.99
    - Class 1: 0.10
- F1-Score:
    - Class 0: 0.96
    - Class 1: 0.17

Classification Report Summary
The model demonstrates high precision and recall for predicting price declines (class 0) but has lower performance in predicting price increases (class 1).

## Limitations
- Data Volatility: The inherent volatility of Bitcoin prices can lead to inaccurate predictions, especially in highly dynamic market conditions.
- Imbalance in Classes: The dataset is imbalanced, with many more instances of price declines compared to increases, affecting the model's ability to predict positive price movements effectively.


## Ethical Considerations
Users should exercise caution and consider the potential financial implications of using this model. Predictions are based on historical data and technical indicators, which may not guarantee future performance.

