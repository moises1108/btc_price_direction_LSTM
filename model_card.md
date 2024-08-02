# Model Card

## Model Description

**Input:** 
Close price, news sentiment data

**Output:** 
Binary price direction prediction for T+1.

**Model Architecture:** 
LSTM model with two LSTM layers, each containing 50 units.
Dropout layers (0.2) are added after each LSTM layer to prevent overfitting.
A final Dense layer with a sigmoid activation function outputs a binary classification (up or down).


## Performance
              precision    recall  f1-score   support

           0       0.93      0.99      0.96      2649
           1       0.55      0.10      0.17       205

    accuracy                           0.93      2854
   macro avg       0.74      0.55      0.57      2854
weighted avg       0.91      0.93      0.91      2854


## Limitations
- Data Volatility: The inherent volatility of Bitcoin prices can lead to inaccurate predictions, especially in highly dynamic market conditions.
- Imbalance in Classes: The dataset is imbalanced, with many more instances of price declines compared to increases, affecting the model's ability to predict positive price movements effectively.

## Trade-offs

Outline any trade-offs of your model, such as any circumstances where the model exhibits performance issues. 

Number of LSTM Units
The number of units was increased to enhance the model’s capacity to learn complex patterns, but this also raised the risk of overfitting and required more computational resources. Conversely, reducing the units led to underfitting.

Number of Layers
Adding layers improved the model's ability to capture hierarchical features, but it also increased complexity and training time. More layers heightened the risk of vanishing gradients and overfitting.

Dropout Rate
A higher dropout rate helped prevent overfitting but sometimes led to underfitting if too many neurons were dropped. Lower rates improved training performance but resulted in a model that generalized poorly.

Learning Rate
A higher learning rate sped up convergence but caused the model to overshoot the optimal solution. A lower learning rate provided more precise convergence but increased training time and risked getting stuck in local minima.

Batch Size
Smaller batch sizes led to better generalization and more stable updates but increased training time. Larger batch sizes accelerated training but caused the model to converge to suboptimal solutions.

Epochs
More epochs improved model performance by allowing more learning, but they also led to overfitting. Fewer epochs sometimes prevented the model from fully learning the underlying patterns in the data.