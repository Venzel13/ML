from config import PARAMS
from learn import predict_ts
from metrics import smape
from preproc import preprocess_data

df = preprocess_data("train.csv")

def subset_data(data, country, state, county):
    ts = (
        data.query("country_region == @country")
        .query("province_state == @state")
        .query("county == @county")
    )
    return ts

ts = subset_data(df, 'US', 'California', 'Los Angeles')

result = predict_ts(ts, ["2020-05-01", "2020-05-22"], PARAMS)
print(result)
result