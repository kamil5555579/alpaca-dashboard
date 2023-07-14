from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
import psycopg2 as ps
import sqlalchemy

app = Dash(__name__)

df = pd.read_csv('data.csv', delimiter=r"\s+", nrows=16, index_col='Run_Number_Run_Number___value')

dropdown_options = [
    {'label': 'Graph 1', 'value': 'graph1'},
    {'label': 'Graph 2', 'value': 'graph2'},
    {'label': 'Graph 3', 'value': 'graph3'}
]

engine=sqlalchemy.create_engine("postgresql+psycopg2://postgres:admin@localhost/alpaca")
with engine.begin() as conn:
    query = sqlalchemy.text('''SELECT html_content FROM graphs
                            WHERE id=1''')
    html_content_1=conn.execute(query).fetchone()[0]

    query = sqlalchemy.text('''SELECT html_content FROM graphs
                            WHERE id=2''')
    html_content_2=conn.execute(query).fetchone()[0]

with open('assets/graph1.html', 'w', encoding='utf-8') as file:
    file.write(html_content_1)

with open('assets/graph2.html', 'w', encoding='utf-8') as file:
    file.write(html_content_2)

app.layout = html.Div([
    html.H1(id='header-title',children='Alpaca', style={'textAlign':'center'}), #header
    dcc.Dropdown(
                options=[{'label': index, 'value': index} for index in df.index],
                value=df.index[0],
                id='dropdown-selection'
            ),
    dash_table.DataTable(columns=[{"name": col, "id": col} for col in df.columns],
            data=[], id='table'),
    html.Div(id='graphs', children=
                [ #graph area
    html.Div(id='graph-container', # 1st col
            children=[
                    html.Div(html.Iframe(id='graph_html_1', src="assets/graph1.html", className='html-graph'),className='html-graph-div'),
                    html.Div(children=[
                                dcc.Dropdown(id='graph-selection', options = dropdown_options, value='graph1'),
                                html.Iframe(id='graph', style={"height": "500px", "width": "100%"})
                                ]
                    )], style={'width': '48%', 'display': 'inline-block'}),
    html.Div(id='graph-container', #2nd col
            children=[
                    html.Div(html.Iframe(id='graph_html_2', className='html-graph', src="assets/graph2.html"),className='html-graph-div'),
                    html.Div(dcc.Graph(id='graph3'))
                    ], style={'width': '48%', 'display': 'inline-block'}),
                ], style={'display': 'flex', 'flex-direction': 'row'}),
    dcc.Interval(
            id='interval-component',
            interval=10*1000, # in milliseconds
            n_intervals=0
        ),
        html.Iframe(id='graph_html_1', style={"height": "515px",
                                                                    "width": "100%",
                                                                    "border": "1px solid #ccc",
                                                                    "border-radius": "50px",
                                                                    "padding": "0px !important",
                                                                    "margin":"0px"},
                                                                      src="assets/graph1.html")
])

@app.callback(
    Output('table', 'data'),
    Input('dropdown-selection', 'value')
)
def update_table(value):
    updated_data = df.loc[value:value].to_dict('records')
    return updated_data

@app.callback(
    Output('graph', 'src'),
    Input('graph-selection', 'value')
)
def update_graph(selected_graph):
    if selected_graph == 'graph1':
        src = "assets/371284.html"
        return src

    elif selected_graph == 'graph2':
        src = "assets/graph.html"
        return src

    elif selected_graph == 'graph3':
        src = "assets/graph1.html"
        return src
    
@app.callback(
    Output("graph3", "figure"),
    Input('interval-component', 'n_intervals')
)
def update_metrics(n_intervals):
    engine=sqlalchemy.create_engine("postgresql+psycopg2://postgres:kamilkamil@database-2.c3fqclqzgkdb.eu-north-1.rds.amazonaws.com/project")
    with engine.begin() as conn:
        query = sqlalchemy.text("""SELECT * FROM videos""")
        df = pd.read_sql_query(query, conn)

    return px.scatter(df, x="upload_date", y="view_count")

    
if __name__ == "__main__":
    app.run_server(debug=True)

