import requests
import json
import os
import yfinance as yf
import torch
from chronos import ChronosPipeline
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

def get_stock_forecast(stock_symbol: str, prediction_days: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Forecasting function for any given ticker

    Args:
        stock_symbol (str): The stock symbol to forecast (e.g., 'AAPL', 'MSFT').
        prediction_days (int, optional): The prediction days. Defaults to 10.

    Returns:
        pd.DataFrame: Dataframe containing: 
            - ticker (string)
            - date (datetime.date)
            - prediction (float)
    """

    stock_data = yf.download(stock_symbol, period="1y", interval="1d")
    
    stock_data = stock_data.reset_index().rename(columns={"Date": "ds", "Close": "y"})
    stock_data = stock_data[["ds", "y"]].copy()
    stock_data.columns = ["ds", "y"]
    stock_data['unique_id'] = stock_symbol

    pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="mps", # configure Mac MPS GPU
    torch_dtype=torch.bfloat16,
    )   

    historical_data = stock_data['y'].tolist()

    context = torch.tensor(historical_data)

    # predict using LLM model by passing context data
    forecasts = pipeline.predict(context, prediction_days)  

    df_forecast = pd.DataFrame()
    for ts_key, forecast in zip([stock_symbol], forecasts):
        low, median, high = np.quantile(forecast.numpy(), [0.1, 0.5, 0.9], axis=0)

        df_forecast = pd.DataFrame({'forecast_lower': low,
                            'forecast_median':median,
                            'forecast_high':high
                            })
        df_forecast['ticker'] = ts_key

    return df_forecast, stock_data


def stock_forecast_agent(prompt: str)-> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Using OpenAI api and GPT4mini predict the stock price 
    of a company available in the Yahoo Finance API for any days in the futures.

    Args:
        prompt (str): the user prompt
    """
        
    api_key = os.environ['OPENAI_API_KEY']
    endpoint = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Define the function schema for OpenAI function calling
    function_schema = {
        "name": "get_stock_forecast",
        "description": "Get stock price forecast for a given number of days",
        "parameters": {
            "type": "object",
            "properties": {
                "stock_symbol": {
                    "type": "string",
                    "description": "The stock symbol to forecast (e.g., 'AAPL', 'MSFT')."
                },
                "prediction_days": {
                    "type": "integer",
                    "description": "The number of days to forecast prices for."
                }
            },
            "required": ["stock_symbol"]
        }
    }

    messages = [
        {"role": "user", "content": prompt}
    ]

    data = {
        "model": "gpt-4o-mini",  
        "messages": messages,
        "functions": [function_schema],
        "function_call": "auto"
    }

    response = requests.post(endpoint, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        result = response.json()
        function_call = result["choices"][0].get("message", {}).get("function_call")

        if function_call:
            arguments = json.loads(function_call["arguments"])
            stock_symbol = arguments["stock_symbol"]
            prediction_days = arguments.get("prediction_days", 10)
            df_forecast, df_historical_data = get_stock_forecast(stock_symbol, prediction_days)
        else:
            print(result["choices"][0]["message"]["content"])
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")

    return df_forecast, df_historical_data, arguments


# if __name__ == '__main__':

#     prompt = "Get stock price forecast for NVIDIA for 15 days"
#     df_forecast, historical_data, arguments = stock_forecast_agent(prompt=prompt)
#     print(df_forecast.head())