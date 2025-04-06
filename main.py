from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = FastAPI(title="Budget Forecast API")

class ForecastRequest(BaseModel):
    data: object = {}


class ForecastResponse(BaseModel):
    forecast: dict
    cushion: float
    last_updated: datetime



@app.post("/forecast", response_model=ForecastResponse)
async def get_forecast(request: ForecastRequest):
    df = pd.read_json('data.json', lines=True)
    df['date'] = pd.to_datetime(df['date'])

    # Агрегируем данные по месяцам
    monthly = (
        df
            .resample('M', on='date')['cost']
            .sum()
            .reset_index()
            .rename(columns={'date': 'Month', 'cost': 'Total'})
    )

    # Добавляем признаки
    monthly['Month_Num'] = np.arange(1, len(monthly) + 1)
    monthly['MonthOfYear'] = monthly['Month'].dt.month
    monthly['Quarter'] = monthly['Month'].dt.quarter

    # Фичи: лаги и скользящие
    monthly['Lag1'] = monthly['Total'].shift(1).fillna(0)
    monthly['Lag2'] = monthly['Total'].shift(2).fillna(0)
    monthly['Lag3'] = monthly['Total'].shift(3).fillna(0)
    monthly['RollMean3'] = monthly['Total'].rolling(3).mean().shift(1).fillna(0)
    monthly['RollMed6'] = monthly['Total'].rolling(6).median().shift(1).fillna(0)


    artifacts = joblib.load('model_artifacts.joblib')
    model = artifacts['model']
    FEATURES = artifacts['features_order']

    prev_totals = monthly['Total'].tolist()
    prev_dates = monthly['Month'].tolist()
    prev_month_nums = monthly['Month_Num'].tolist()
    prev_month_of_year = monthly['MonthOfYear'].tolist()

    forecast = []
    horizon = 3

    for i in range(horizon):
        mn = prev_month_nums[-1] + 1
        moy = (prev_month_of_year[-1] % 12) + 1
        q = ((moy - 1) // 3) + 1
        lag1 = prev_totals[-1]
        lag2 = prev_totals[-2]
        lag3 = prev_totals[-3]
        roll3 = np.mean(prev_totals[-3:])
        roll6 = np.median(prev_totals[-6:]) if len(prev_totals) >= 6 else np.median(prev_totals)

        Xp = [[mn, moy, q, lag1, lag2, lag3, roll3, roll6]]
        pred = model.predict(Xp)[0]

        next_date = pd.date_range(prev_dates[-1], periods=2, freq='M')[1]
        forecast.append({'Month': next_date, 'Forecast': pred})

        prev_dates.append(next_date)
        prev_month_nums.append(mn)
        prev_month_of_year.append(moy)
        prev_totals.append(pred)

    forecast_df = pd.DataFrame(forecast)

    cushion_3m = forecast_df['Forecast'].sum()
    # Преобразуем столбец 'Month' в строки с нужным форматом
    forecast_df['Month'] = pd.to_datetime(forecast_df['Month']).dt.strftime('%Y-%m-%d')

    # Преобразуем в формат {"month": forecast}
    result = forecast_df.set_index('Month').to_dict()['Forecast']

    return {
        'forecast': result,
        'cushion': cushion_3m,
        'last_updated': datetime.now()
    }

