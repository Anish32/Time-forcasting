def app():
    st.title("📈 Stock Price Prediction App")

    stock = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31"))

    model_choice = st.selectbox("Choose Model", ["GRU", "ARIMA"])

    if st.button("Predict"):
        df = yf.download(stock, start=start_date, end=end_date)
        st.write("### Raw Data", df.tail())
        st.line_chart(df['Close'])

        if model_choice == "GRU":
            st.write("### Running GRU Model...")
            data = df[['Close']]
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data)

            def create_sequences(data, step=60):
                X, y = [], []
                for i in range(len(data) - step - 1):
                    X.append(data[i:(i + step), 0])
                    y.append(data[i + step, 0])
                return np.array(X), np.array(y)

            X, y = create_sequences(scaled)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            model = Sequential()
            model.add(GRU(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(GRU(50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

            pred = model.predict(X_test)
            pred_rescaled = scaler.inverse_transform(pred)
            actual_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

            st.line_chart(pd.DataFrame({
                "Actual": actual_rescaled.flatten(),
                "Predicted": pred_rescaled.flatten()
            }))

        elif model_choice == "ARIMA":
            st.write("### Running ARIMA Model...")
            close_data = df[['Close']].dropna()
            train_size = int(len(close_data) * 0.8)
            train, test = close_data[:train_size], close_data[train_size:]
            model = ARIMA(train, order=(5, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(test))
            forecast.index = test.index

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(train.index, train['Close'], label='Train')
            ax.plot(test.index, test['Close'], label='Actual')
            ax.plot(forecast.index, forecast, label='ARIMA Forecast')
            ax.legend()
            st.pyplot(fig)

            arima_df = pd.DataFrame({
                'Actual': test['Close'].values,
                'ARIMA Forecast': forecast.values
            }, index=test.index)

            st.download_button("Download ARIMA Results", arima_df.to_csv(index=True), "arima_predictions.csv")
