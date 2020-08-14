import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

from config import PARAMS
from learn import predict_ts
from metrics import smape
from preproc import preprocess_data


df = preprocess_data("train.csv")

style = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
size = 100
app = dash.Dash(__name__, external_stylesheets=style)


def subset_data(data, country, state, county):
    ts = (
        data.query("country_region == @country")
            .query("province_state == @state")
            .query("county == @county")
    )
    return ts


def create_options(data, granularity):
    return [
        {"label": str(location), "value": location}
        for location in data[granularity].unique()
    ]


app.layout = html.Div(
    [
        html.H1("Wine dataset", style={"color": "red", "textAlign": "center"}),
        html.Div(id="aaa", children="My attempt to make a custom graph"),
        dcc.Graph(id="graph"),
        dcc.Dropdown(
            id="dropdown_country",
            options=create_options(df, "country_region"),
            placeholder="country",
            value="US",
        ),
        dcc.Dropdown(id="dropdown_state", placeholder="state",),
        dcc.Dropdown(id="dropdown_county", placeholder="county",),
        html.Button(id="button", n_clicks=0, children="press"),
    ]
)


@app.callback(
    Output("dropdown_state", "options"),
    [Input("dropdown_country", "value")]
)
def update_city(country):
    sub_df = df.query("country_region == @country")
    state = create_options(sub_df, "province_state")
    return state


@app.callback(
    Output("dropdown_county", "options"),
    [Input("dropdown_state", "value")]
)
def update_county(state):
    sub_df = df.query("province_state == @state")
    county = create_options(sub_df, "county")
    return county


@app.callback(
    Output("graph", "figure"),
    [Input("button", "n_clicks")],
    [
        State("dropdown_country", "value"),
        State("dropdown_state", "value"),
        State("dropdown_county", "value"),
    ],
)
def hey(n_clicks, country, state, county):
    ts = subset_data(df, country, state, county)
    result = predict_ts(ts, ["2020-05-01", "2020-05-22"], PARAMS)

    line_chart = go.Figure()
    line_chart.add_trace(
        go.Scatter(
            x=result["date"], y=result["infected"], mode="lines", name="Infected"
        )
    )
    line_chart.add_trace(
        go.Scatter(
            x=result["date"], y=result["pred_infected"], mode="lines", name="Predicted"
        )
    )

    return line_chart


if __name__ == "__main__":
    app.run_server(debug=False)
