from dash import Dash, html, dash_table, dcc, callback, Output, Input, State, MATCH
import plotly.express as px
import pandas as pd
import psycopg2 as ps
import sqlalchemy
import re
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
import dash_draggable
import copy

app = Dash(__name__)

# function that selects all the data from alpaca database
def fetch_data():

    engine=sqlalchemy.create_engine("postgresql+psycopg2://postgres:admin@localhost/alpaca")
    with engine.begin() as conn:
        query = sqlalchemy.text("""SELECT * FROM alpaca
                                ORDER BY "Run_Number_Run_Number___value" DESC
                                """)
        df = pd.read_sql_query(query, conn)

    df.set_index('Run_Number_Run_Number___value', inplace=True)

    # need to convert arrays saved as bytes back to arrays
    byte_columns = [column for column in df.columns if column.endswith("_shape") is False and df[column].dtype == object]
    shape_columns = [f'{column}_shape' for column in byte_columns]
    image_columns =[]

    for shape_column in shape_columns:
        df[shape_column] = df[shape_column].apply(lambda x: tuple(map(int, re.findall(r'\d+', x))) if pd.notnull(x) else None)  # Extract the shape values


    for column in byte_columns:
        shape_column = f'{column}_shape'
        df[column] = df.apply(lambda row: np.frombuffer(row[column]).reshape(row[shape_column]) if pd.notnull(row[column]) and pd.notnull(row[shape_column]) else None, axis=1)

    return df, byte_columns

# plotting functions taken from alpaca

def scatter_fig(df, x_col, y_col):
        
    # erase NaN values
    df = df[df[y_col].notnull()]

    # make arrays
    x = np.array(df[x_col])
    y = np.array(df[y_col])

    fig = dict({
            "data": [{"type": 'scatter',
                      "x": x,
                      "y": y,
                      "mode": 'markers',  # lines, markers
                      "marker": {"color": '#FFFFFF', "symbol": 'circle', "size": 10, "opacity": 0.5,
                                 # default marker
                                 "line": {"color": 'Black', "width": 1.5}},
                      "name": y_col,
                      "showlegend": False}  # default: False
                     ],
            "layout": {"barmode": "stack", "title": {"text": y_col, "font": {"size": 20}},
                       "legend": {"bgcolor": '#FFFFFF', "bordercolor": '#ff0000', "font": {"size": 25},
                                  "orientation": 'v'},
                       "xaxis": {"title": {"text": x_col, "font": {"size": 15}},
                                 "tickfont": {"size": 15},
                                 "autorange": True, "fixedrange": False, "type": 'linear', "gridcolor": "black",
                                 "linecolor": "black", "linewidth": 4, "ticks": "inside", "tickwidth": 5,
                                 "nticks": 20,
                                 "ticklabelstep": 2, "ticklen": 10, "position": 0, "mirror": "all"},
                       "yaxis": {
                                 "tickfont": {"size": 15},
                                 "autorange": True, "fixedrange": False, "exponentformat": "power",
                                 "gridcolor": "black", "linecolor": "black", "linewidth": 4, "ticks": "inside",
                                 "tickwidth": 5, "nticks": 20, "ticklabelstep": 2, "ticklen": 10,
                                 "mirror": "allticks"}}
        })

    return fig

def surface_fig(df, x_col, y_col, z_col):

    # erase NaN values
    df = df[df[z_col].notnull()]

    # make arrays
    x = np.array(df[x_col])
    y = np.array(df[y_col])
    z = np.array(df[z_col])

    # make linear spaces
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)

    # make meshgrid
    X, Y = np.meshgrid(xi, yi)

    # add interpolation
    Z = griddata((x, y), z, (X, Y), method='linear')  # {‘linear’, ‘nearest’, ‘cubic’}

    # create the dictionary for the surface
    fig = dict({
            "data": [{"type": 'surface',
                      "x": X,
                      "y": Y,
                      "z": Z,
                      "connectgaps": True,
                      "opacity": 1,
                      # "colorscale": 'Blues',
                      "colorbar": {"title": {"text": "MCP integral" + '<br>' + "[arb. units]", "side": 'top'},
                                   "tickfont": {"size": 16}, 'lenmode': 'fraction', 'len': 0.6,
                                   'showexponent': 'none'},
                      "text": [x_col, y_col, z_col],
                      "contours_z": dict(show=True, usecolormap=True, highlightcolor="black", project_z=True)
                      }],

            "layout": {"scene": {
                "xaxis": {"title": {"text": x_col, "font": {"size": 10}}, "tickfont": {"size": 14},
                          "linewidth": 0, "ticks": 'outside'},
                "yaxis": {"title": {"text": y_col, "font": {"size": 10}}, "tickfont": {"size": 14},
                          "linewidth": 0, "ticks": 'outside'},
                # 'showexponent': 'last', 'exponentformat': 'none', "ticksuffix": '', range: [-16, 8]
                "zaxis": {'showexponent': 'none', "ticks": 'outside', "tickangle": 0,
                          "title": {"text": z_col, "font": {"size": 10}}, "tickfont": {"size": 14},
                          "linewidth": 0}},  # "title": self.title,
                       "margin": dict(l=20, r=20, t=20, b=20),
                       "legend": {"bgcolor": '#FFFFFF', "bordercolor": '#ff0000', "font": {"size": 25},
                                  "orientation": 'v'},
                       "showlegend": True}
        })
    return fig

def histogram_fig(df, x_col, y_col):

    # erase NaN values
    df = df[df[y_col].notnull()]

    # make arrays
    x = np.array(df[x_col])
    y = np.array(df[y_col])

    fig = dict({
            "data": [{"type": 'histogram',
                      "x": x,
                      "y": y,
                      "histfunc": "count",
                      "name": y_col,
                      "orientation": "v",
                      "textfont": {"size": 45},
                      "xbins": {"size": 0.1}  # Scintillator time resolution: 0.0000001
                      }],
            "layout": {"xaxis": {"title": x_col},
                       "yaxis": {"title": y_col},
                       # "yaxis_range": [0, 300],
                       # "xaxis_range": [17, 33],

                       "legend": {"bgcolor": '#FFFFFF', "bordercolor": '#ff0000', "font": {"size": 25},
                                  "orientation": 'v'},
                       "showlegend": False}
        })
    return fig

df, byte_columns = fetch_data()
numerical_columns = df.select_dtypes(include='number').columns

# dropdowns options
run_options = [{'label': number, 'value': number} for number in df.index.values]
image_options = [{'label': col, 'value': col} for col in byte_columns]
numerical_variable_options = [{'label': col, 'value': col} for col in numerical_columns]
chart_types = ['3D Plot', 'Image Plot', 'Histogram', 'Scatter Plot']

dummy_image = px.imshow([[0]], color_continuous_scale='gray')

def create_chart_element(type, n_clicks):
    if type=='3D Plot': 
        new_element = html.Div(
            children=[
                dcc.Dropdown(id={'type':'3d-x-selection',
                                 'index':n_clicks}, options=numerical_variable_options),
                dcc.Dropdown(id={'type':'3d-y-selection',
                                 'index':n_clicks}, options=numerical_variable_options),
                dcc.Dropdown(id={'type':'3d-z-selection',
                                 'index':n_clicks}, options=numerical_variable_options),
                dcc.Graph(id={'type':'3d-plot',
                                 'index':n_clicks}, className='graph', figure=go.Figure())
            ],
            className='graph-div'
        )
    if type=='Histogram': 
        new_element = html.Div(
            children=[
                dcc.Dropdown(id='histogram-x-selection', options=numerical_variable_options),
                dcc.Dropdown(id='histogram-y-selection', options=numerical_variable_options),
                dcc.Graph(id='histogram', className='graph', figure=go.Figure())
            ],
            className='graph-div'
        )
    if type=='Image Plot': 
        new_element = html.Div(
            children=[
                dcc.Dropdown(id='run-selection', options=run_options),
                dcc.Dropdown(id='image-selection', options=image_options),
                dcc.Graph(id='image-plot', className='graph', figure=dummy_image)
            ],
            className='graph-div'
        )
    if type=='Scatter Plot': 
        new_element = html.Div(
            children=[
                dcc.Dropdown(id='scatter-x-selection', options=numerical_variable_options),
                dcc.Dropdown(id='scatter-y-selection', options=numerical_variable_options),
                dcc.Graph(id='scatter-plot', className='graph', figure=go.Figure())
            ],
            className='graph-div'
        )
    return new_element

app.layout = html.Div([
    html.Div(children=[
        html.Img(src='assets/alpaca_logo.png'),
        html.H1(id='header-title', children=['Alpaca',html.Br(),'Dashboard']),
        html.Div(id='data-div', children=[dcc.Dropdown(id='chart-type-dropdown', 
                               options=[{'label': chart_type, 'value': chart_type} for chart_type in chart_types],
                                placeholder='Select Chart Type'), html.Br(),
                                html.Button('Add Chart', id='add-chart-btn', n_clicks=0), html.Br(),
                                html.Button('save pos', id='save-button', n_clicks=0)]),
        html.H2(children=[
            html.Img(src='assets/AEgIS-logo.png', id='aegis-logo'), html.Br(),
            'Last run:', df.index[0],html.Br(),
            'check for new runs',html.Br(),
            html.Button('Refresh Data', id='refresh-button')],
            id='right-header')
        ], id='header-area'),
        dash_draggable.ResponsiveGridLayout(id='draggable', children=[])
            ])

# image plot callback

@app.callback(
    Output('image-plot', 'figure', allow_duplicate=True),
    [Input('run-selection', 'value'),
    Input('image-selection', 'value')],
    prevent_initial_call=True,
    pattern='^image-plot-.+'
)
def update_graph(selected_run, selected_img):

    if selected_run and selected_img:
        img = px.imshow(df[selected_img][selected_run], title=byte_columns[0], zmin=0, zmax=np.max(df[selected_img][selected_run]) / 3,  #
                            color_continuous_scale='GnBu')
        return img
    else:
        return px.imshow([[0]], color_continuous_scale='gray')
    
# 3D plot callback

@app.callback(
    Output({'type':'3d-plot', 'index':MATCH}, 'figure', allow_duplicate=True),
    [Input({'type':'3d-x-selection', 'index':MATCH}, 'value'),
    Input({'type':'3d-y-selection', 'index':MATCH}, 'value'),
    Input({'type':'3d-z-selection', 'index':MATCH}, 'value')],
    prevent_initial_call=True)

def update_graph(x,y,z):

    if x and y and z and (x != y):
        fig = surface_fig(df,x,y,z)
        return fig
    else:
        return go.Figure()

# scatter plot callback

@app.callback(
    Output('scatter-plot', 'figure', allow_duplicate=True),
    [Input('scatter-x-selection', 'value'),
    Input('scatter-y-selection', 'value')],
    prevent_initial_call=True,
    pattern='^scatter-plot-.+'
)
def update_graph(x,y):

    if x and y:
        fig = scatter_fig(df,x,y)
        return fig
    else:
        return go.Figure()

# histogram callback

@app.callback(
    Output('histogram', 'figure', allow_duplicate=True),
    [Input('histogram-x-selection', 'value'),
    Input('histogram-y-selection', 'value')],
    prevent_initial_call=True,
    pattern='^histogram-.+'
)
def update_graph(x,y):

    if x and y:
        fig = histogram_fig(df,x,y)
        return fig
    else:
        return go.Figure()

# refresh button callback

@app.callback(
    Output("image-plot", "figure"),
    Input('refresh-button', 'n_clicks'),
    [State('run-selection', 'value'),
    State('image-selection', 'value')],
    prevent_initial_call=True
)
def update_metrics(n_clicks, selected_run, selected_img):

    if n_clicks:
        df, byte_columns = fetch_data()

    if selected_run and selected_img:
        img = px.imshow(df[selected_img][selected_run], title=byte_columns[0], zmin=0, zmax=np.max(df[selected_img][selected_run]) / 3,  #
                            color_continuous_scale='GnBu')
        return img
    else:
        return px.imshow([[0]], color_continuous_scale='gray')

# new chart callback

@app.callback(
    Output('draggable', 'children'),
    State('chart-type-dropdown', 'value'),
    Input('add-chart-btn', 'n_clicks'),
    State('draggable', 'children')
)
def add_chart(chart_type, n_clicks, existing_children):
    if chart_type is None:
        return existing_children

    new_chart = create_chart_element(type=chart_type, n_clicks=n_clicks)
    existing_children.append(new_chart)
    
    return existing_children
    
# save pos callback

@app.callback(
    Output('data-div', 'children'),
    Input('save-button', 'n_clicks'),
    State('draggable', 'layouts'),
    prevent_initial_call=True
)
def save_position(n_clicks, layout):
    
    print(layout)
    return None
    

if __name__ == "__main__":
    app.run_server(debug=True)

