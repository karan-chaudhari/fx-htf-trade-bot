# fx-htf-trade-bot

## Flowchart of the MLIndicatorCalculator Model

This section outlines the flow of data and the processes involved in the `MLIndicatorCalculator` model used for trading signal predictions.

### Flow Overview

1. **Start**
   - The process begins here.

2. **Data Input**
   - Inputs: `high`, `low`, `close_prices`.

3. **Prepare Features**
   - Calls the `prepare_features(high, low, close_prices)` method:
     - Checks if there are enough `close_prices`.
     - Calculates various indicators: SMA, RSI, MACD, ATR, and Stochastic.
     - Combines the calculated indicators into a DataFrame.
     - Creates a target variable based on future price movements.
     - Returns the features and target.

4. **Train Model**
   - Calls the `train_model(high, low, close_prices)` method:
     - Utilizes `prepare_features()` to gather necessary data.
     - Scales features using `StandardScaler`.
     - Performs `GridSearchCV` for hyperparameter tuning.
     - Selects the best estimator based on performance.
     - Splits the data into training and testing sets.
     - Evaluates model accuracy.
     - Logs results and updates the `is_model_trained` flag based on accuracy.

5. **Predict Signal**
   - Calls the `predict_signal(high, low, close_prices)` method:
     - Checks if the model is trained and ready for predictions.
     - Calls `prepare_features()` to obtain features for prediction.
     - Scales the features.
     - Makes predictions based on the model.
     - Logs and returns the predicted signal: "buy", "sell", or "trade not allowed".

6. **End**
   - The process concludes here.

### Flowchart Visualization

```plaintext
[Start]
   |
[Data Input]
   |
[Prepare Features]
   |
[Train Model]-------------------> [Predict Signal]
   |                                     |
[End] <---------------------------------|
