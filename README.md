An AI-driven web application that forecasts **Ethereum (ETH)** prices using a **Long Short-Term Memory (LSTM)** neural network. The project features a **Flask backend** that fetches historical price data from Yahoo Finance, preprocesses it, and uses a **pre-trained LSTM model** to predict future prices. A **React frontend** consumes the backend API to display real-time predictions in a modern, responsive UI.

- 🧠 **Deep Learning Forecasting**
  - Uses an LSTM model trained on historical ETH prices to generate predictions.
- 🔄 **Live Data from Yahoo Finance**
  - Auto-fetches the latest historical prices using the `yfinance` library.
- 🧹 **Efficient Preprocessing**
  - Normalizes price data and reshapes it into sequences suitable for time-series prediction.
- 🧪 **RESTful Flask API**
  - Predict prices and fetch real-time or historical data via HTTP endpoints.
- 🌐 **Modern React Dashboard**
  - Visualize predictions with real-time charts and responsive design.
- 📈 **Interactive Charts**
  - Displays real and forecasted prices using Chart.js or Recharts.
- 🧭 **Prediction Timeframes**
  - Supports flexible lookahead options (e.g., 1-day, 3-day, 7-day predictions).
- 📉 **Seamless UX**
  - Minimal UI with elegant transitions, input forms, and loading states.


```bash
eth-price-predictor/
├── backend/
│   ├── app.py                 # Flask API server
│   ├── model/
│   │   └── eth_lstm_model.h5  # Pretrained LSTM model
│   ├── utils/
│   │   ├── fetch_data.py      # Fetch data from yfinance
│   │   └── preprocess.py      # Preprocess and normalize data
│   └── requirements.txt       # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/        # React components (Chart, Header, etc.)
│   │   ├── services/          # Axios API requests
│   │   ├── App.js             # Root component
│   │   └── index.js           # Entry point
│   └── package.json           # Node dependencies
├── README.md
