import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PARAMS
from learn import predict_ts
from metrics import smape
from preproc import preprocess_data


df = preprocess_data("train.csv")



ts = (
    df.query("country_region == 'US'")
    .query("province_state == 'None'")
    .query("county == 'None'")
)



result = predict_ts(ts, ["2020-05-01", "2020-05-22"], PARAMS)
result