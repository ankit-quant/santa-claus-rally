import pandas as pd
import numpy as np
import os
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from optimisations.optimisations.database import local as local_db


def resample_daily(df_min: pd.DataFrame):
    df = df_min.copy()
    df = df.resample('D').agg(
        {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    return df


def analyze_santa_clauss_rally(data: pd.DataFrame, dec_days, jan_days):
    df = data.copy()
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day

    rally_results = []

    for year in range(df['Year'].min(), df['Year'].max()):
        december_data = df[(df['Year'] == year) & (df['Month'] == 12)]
        january_data = df[(df['Year'] == year + 1) & (df['Month'] == 1)]

        if not december_data.empty and not january_data.empty:
            december_rally = december_data.tail(dec_days)
            january_rally = january_data.head(jan_days)

            santa_claus_rally_period = pd.concat(
                [december_rally, january_rally])

            start_price = santa_claus_rally_period.iloc[0]['Close']
            end_price = santa_claus_rally_period.iloc[-1]['Close']
            rally_return = (end_price - start_price) / start_price
            rally_results.append((year, rally_return))
    return pd.DataFrame(rally_results, columns=['Year', 'Return'])


def results_output(data: pd.DataFrame, index_name, dec_days, jan_days):
    df = data.copy()
    results = analyze_santa_clauss_rally(df, dec_days, jan_days)
    average_return = results['Return'].mean()
    positive_rallies = results[results['Return'] > 0]
    percentage_positive = len(positive_rallies) / len(results) * 100
    total_return = (results['Return'].sum()) * 100
    print(f"{index_name} - Avg Return: {100*average_return:.2f}%, Positive(%): {percentage_positive:.2f}, Total Return: {total_return:.2f}% Dec Days:{dec_days},Jan Days:{jan_days}")
    print(results)
    plot_returns(results, index_name, percentage_positive,
                 average_return, dec_days, jan_days)

def calculate_cagr(start_value, end_value, periods):
    return (end_value / start_value) ** (1 / periods) - 1

def plot_returns(data: pd.DataFrame, index_name, win_rate, avg_ret, dec_days, jan_days):
    path = r"path"
    file_name = f"{index_name}_{dec_days}_{jan_days}.png"

    cumulative_returns = 100*((1 + data['Return']).cumprod() - 1)
    # Calculate total return and CAGR
    total_return = cumulative_returns.iloc[-1]
    start_year = data['Year'].iloc[0]
    end_year = data['Year'].iloc[-1]

    periods = end_year - start_year + 1
    cagr = calculate_cagr(
        1, cumulative_returns.iloc[-1]*0.01 + 1, periods) * 100

    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Year', y=cumulative_returns, data=data, marker='o')
    plt.title(
        f'{index_name} - Santa Claus Rally. \n Long {dec_days} Days before end of year. Exit {jan_days} Days in Jan \nTotal Return: {total_return:.2f}%, CAGR: {cagr:.2f}%\nWin Rate:{win_rate:.2f}%,Avg. Return:{100*avg_ret:.2f}%')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Return (%)')

    plt.savefig(os.path.join(path, file_name))
    # plt.show()


if __name__ == "__main__":
    df_bn = local_db.get_raw_index("BANKNIFTY")
    df_nf = local_db.get_raw_index("NIFTY")

    bn_daily = resample_daily(df_bn)
    nf_daily = resample_daily(df_nf)

    dec_days_list = [6, 5, 4]
    jan_days_list = [3, 2, 1]
    for dec_day in dec_days_list:
        for jan_day in jan_days_list:
            # Analyze the Santa Claus Rally
            results_output(bn_daily, index_name="BANKNIFTY",
                           dec_days=dec_day, jan_days=jan_day)
            results_output(nf_daily, index_name="NIFTY",
                           dec_days=dec_day, jan_days=jan_day)
