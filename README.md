# 📊 Stock Price Forecaster using AI Agent & Streamlit

This is a user-friendly Streamlit application that integrates AI-based probabilistic forecasting to predict future stock prices. Powered by Hugging Face's Chronos model and Yahoo Finance, it provides interactive visualizations using Plotly, making it a powerful tool for investors and data enthusiasts.

---

## 🌟 Features

1. **Historical Stock Data Visualization**:
   - Fetches and displays the past year's price and volume data for any stock ticker.

2. **AI-Powered Forecasting**:
   - Uses the Chronos model from Hugging Face to generate probabilistic forecasts.
   - Forecast includes **median, low, and high price estimates**.
   
3. **AI Agent**
   - uses ChatGPT-4-mini as AI Agent to make a function calling on the AWS Chronos model, which generates probabilistic forecasting.
   - Allows the user to write in natural language which forecast to generate, i.g. any company available in yahoo API and any time horizon. 
   - User does not need to know the ticker, the AI Agent finds the correct ticker of the company and calls the API. 

4. **Interactive Visualization**:
   - Historical prices are shown as a line chart.
   - Forecast ranges (low, median, high) are displayed as an interactive Plotly chart with shaded regions.


---

## 🛠️ Installation and Setup

### Prerequisites
- Python 3.8 or later
- [Streamlit](https://streamlit.io/) installed
- [Yahoo Finance API](https://pypi.org/project/yfinance/) for stock data
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) library for Chronos model
- Plotly for interactive visualization
- OPENAI_API_KEY, get yours from [OpenAI Plattform](https://platform.openai.com/)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock-price-forecaster.git
   cd stock-price-forecaster
2. Install dependencies 
    ```
    pip install -r requirements.txt
    ```
3. Run the App 
    ```
    streamlit run app.py
    ```

# 🚀 Usage

1.	Enter a prompt in natural language like : *Get stock price forecast for NVIDIA for the next 15 days*
3.	Press the Generate Forecast button.
4.	View:
    - Historical stock price data.
    - Forecasted prices with a shaded region indicating uncertainty (low to high estimates)