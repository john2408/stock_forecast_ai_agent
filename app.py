import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import torch
from chronos import ChronosPipeline
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from src.forecast_agent import stock_forecast_agent, get_stock_forecast


if __name__ == '__main__':

    st.title("Chat-Based AI Stock Forecast Agent App")
    st.write("Enter your query as a prompt, e.g., 'Get stock price forecast for NVIDIA for the next 15 days'.")

    user_prompt = st.text_input("Enter your prompt:", "")

    if user_prompt:
        try:
            forecast, df_historical_data, arguments = stock_forecast_agent(prompt=user_prompt)
            stock_symbol = arguments["stock_symbol"]
            prediction_days = arguments.get("prediction_days", 10)

            st.write(f"Ready to fetch forecast for {stock_symbol} for {prediction_days} days.")
            
            # Add a button to generate the forecast
            if st.button("Generate Forecast"):

                st.write(f"Generating forecast for {stock_symbol} with Chronos Model...")
                

                st.write(f"Forecast for {stock_symbol} for next {prediction_days} prediction_days")
                

                # Plot historical and forecast data
                historical_horizon = 100
                historical_dates = df_historical_data['ds'].values[-historical_horizon:]
                historical_prices = df_historical_data['y'].values[-historical_horizon:]

                forecast_dates = pd.date_range(
                    start=historical_dates[-1] + pd.Timedelta(days=1), periods=prediction_days
                )

                fig = go.Figure()

                # Add historical data
                fig.add_trace(go.Scatter(
                    x=historical_dates,
                    y=historical_prices,
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='blue')
                ))

                # Add forecast median
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast["forecast_median"],
                    mode='lines',
                    name='Median Forecast',
                    line=dict(color='orange')
                ))

                # Add forecast range (low-high) as a filled area
                fig.add_trace(go.Scatter(
                    x=forecast_dates.tolist() + forecast_dates[::-1].tolist(),
                    y=forecast["forecast_high"].tolist() + forecast["forecast_lower"][::-1].tolist(),
                    fill='toself',
                    name='Forecast Range (Low-High)',
                    fillcolor='rgba(255, 165, 0, 0.3)',
                    line=dict(color='rgba(255, 165, 0, 0)')
                ))

                # Customize the layout
                fig.update_layout(
                    title=f"{stock_symbol} Historical Data and Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    template="plotly_white"
                )

                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(forecast)
        except Exception as e:
            st.error(f"Error processing the forecast: {e}")