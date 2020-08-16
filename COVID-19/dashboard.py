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
        html.H1("COVID-19"),
        html.H2(children="Covid forecaster for the different country/states"),
        html.Div("Choose the country and region of interest"),
        html.Div(
            [
                dcc.Dropdown(
                    id="dropdown_country",
                    className="dropdown",
                    options=create_options(df, "country_region"),
                    placeholder="country",
                    value="US",
                ),
            ],
            className="selector",
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id="dropdown_state", placeholder="state", className="dropdown"
                )
            ],
            className="selector",
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id="dropdown_county", placeholder="county", className="dropdown"
                )
            ],
            className="selector",
        ),
        html.Div(html.Button(id="button", n_clicks=0, children="Predict")),
        html.Div(
            [
                html.Div("The value of sMAPE:  ", className='selector'),
                html.Div(id="metric", className='selector')
            ],
        ),
        dcc.Graph(id="graph"),
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
    [Output("metric", "children"), Output("graph", "figure")],
    [Input("button", "n_clicks")],
    [
        State("dropdown_country", "value"),
        State("dropdown_state", "value"),
        State("dropdown_county", "value"),
    ],
)
def plot_graph_metric(n_clicks, country, state, county):
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

    metric = result.query("date >= '2020-05-22'")
    metric = smape(metric["infected"].values, metric["pred_infected"].values)

    return round(metric, 2), line_chart


if __name__ == "__main__":
    app.run_server(debug=False)