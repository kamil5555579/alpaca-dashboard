from dash import Dash, html, dash_table, dcc, callback, Output, Input, State, MATCH, ALL, Patch, ctx, callback_context
import plotly.express as px
import pandas as pd
import psycopg2 as ps
import sqlalchemy
import re
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
import dash_draggable
import json
import dash_daq as daq
from dash.exceptions import PreventUpdate

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
chart_types = ['3d-plot', 'image-plot', 'histogram', 'scatter-plot']

dummy_image = px.imshow([[0]], color_continuous_scale='gray')

def create_chart_element(type, n_clicks, x_val=None, y_val=None, z_val=None, run=None, image=None):
    if type=='3d-plot': 
        new_element = html.Div(
            children=[
                html.Button('X', id={'type':'close-button',
                                 'index':n_clicks}, n_clicks=0),
                dcc.Dropdown(id={'type':'3d-x-selection',
                                 'index':n_clicks}, options=numerical_variable_options, value=x_val),
                dcc.Dropdown(id={'type':'3d-y-selection',
                                 'index':n_clicks}, options=numerical_variable_options, value=y_val),
                dcc.Dropdown(id={'type':'3d-z-selection',
                                 'index':n_clicks}, options=numerical_variable_options, value=z_val),
                dcc.Graph(id={'type':'3d-plot',
                                 'index':n_clicks}, className='graph', figure=go.Figure())
            ],
            className='graph-div'
        )
    if type=='histogram': 
        new_element = html.Div(
            children=[
                html.Button('X', id={'type':'close-button',
                                 'index':n_clicks}, n_clicks=0),
                dcc.Dropdown(id={'type':'histogram-x-selection',
                                 'index':n_clicks}, options=numerical_variable_options, value=x_val),
                dcc.Dropdown(id={'type':'histogram-y-selection',
                                 'index':n_clicks}, options=numerical_variable_options, value=y_val),
                dcc.Graph(id={'type':'histogram',
                                 'index':n_clicks}, className='graph', figure=go.Figure())
            ],
            className='graph-div'
        )
    if type=='image-plot': 
        new_element = html.Div(
            children=[
                html.Button('X', id={'type':'close-button',
                                 'index':n_clicks}, n_clicks=0),
                dcc.Dropdown(id={'type':'run-selection',
                                 'index':n_clicks}, options=run_options, value=run),
                dcc.Dropdown(id={'type':'image-selection',
                                 'index':n_clicks}, options=image_options, value=image),
                dcc.Graph(id={'type':'image-plot',
                                 'index':n_clicks}, className='graph', figure=dummy_image)
            ],
            className='graph-div'
        )
    if type=='scatter-plot': 
        new_element = html.Div(
            children=[
                html.Button('X', id={'type':'close-button',
                                 'index':n_clicks}, n_clicks=0),
                dcc.Dropdown(id={'type':'scatter-x-selection',
                                 'index':n_clicks}, options=numerical_variable_options, value=x_val),
                dcc.Dropdown(id={'type':'scatter-y-selection',
                                 'index':n_clicks}, options=numerical_variable_options, value=y_val),
                dcc.Graph(id={'type':'scatter-plot',
                                 'index':n_clicks}, className='graph', figure=go.Figure())
            ],
            className='graph-div'
        )
    return new_element

app.layout = html.Div([
    html.Div(children=[
        html.Img(src='assets/alpaca_logo.png'),
        html.Div(id='mid-header', children=[html.H1(id='header-title', children=['Alpaca',html.Br(),'Dashboard']), html.Br(),
                                            html.Div([dcc.Dropdown(id='chart-type-dropdown', 
                                            options=[{'label': chart_type, 'value': chart_type} for chart_type in chart_types],
                                            placeholder='Select Chart Type', style={'width': '200px'}),
                                            html.Button('Add Chart', id='add-chart-btn', n_clicks=0)], className='row'), html.Br(),
                                            html.Div([dcc.Dropdown(id='dashboard-name-dropdown',
                                            placeholder='Select user dash', style={'width': '200px'}),
                                            html.Button('save pos', id='save-button', n_clicks=0, type='button'),
                                            html.Button('load', id='load-button', n_clicks=0)], className='row'), html.Br(),
                                            html.Div([dcc.Input(id='new-dashboard', 
                                            type='text',
                                            placeholder='create new dash', style={'width': '200px'}),
                                            html.Button('save', id='new-dash-button', n_clicks=0, type='button')], className='row')]),
        html.H2(children=[
            html.Img(src='assets/AEgIS-logo.png', id='aegis-logo'), html.Br(),
            'Last run:', df.index[0],html.Br(),
            'check for new runs',html.Br(),
            daq.ToggleSwitch(id='my-toggle-switch', value=False), html.Br(),
            html.Div(id='my-toggle-switch-output'),
            dcc.Interval(id='interval-component', interval=5*1000, n_intervals=0)],
            id='right-header')
        ], id='header-area'),
        dash_draggable.ResponsiveGridLayout(id='draggable', children=[]),
        html.Div(id='dummy-out')
            ])

# image plot callback

@app.callback(
    Output({'type':'image-plot', 'index':MATCH}, 'figure'),
    [Input({'type':'run-selection', 'index':MATCH}, 'value'),
    Input({'type':'image-selection', 'index':MATCH}, 'value')])
def update_graph(selected_run, selected_img):

    if selected_run and selected_img:
        data = df[selected_img][selected_run]
        if data is not None and data.any():
            img = px.imshow(df[selected_img][selected_run], title=byte_columns[0], zmin=0, zmax=np.max(df[selected_img][selected_run]) / 3,  #
                            color_continuous_scale='GnBu')
            return img
        else:
            return px.imshow([[0]], color_continuous_scale='gray')
    else:
        return px.imshow([[0]], color_continuous_scale='gray')
    
# 3D plot callback

@app.callback(
    Output({'type':'3d-plot', 'index':MATCH}, 'figure'),
    [Input({'type':'3d-x-selection', 'index':MATCH}, 'value'),
    Input({'type':'3d-y-selection', 'index':MATCH}, 'value'),
    Input({'type':'3d-z-selection', 'index':MATCH}, 'value')])

def update_graph(x,y,z):

    if x and y and z and (x != y):
        fig = surface_fig(df,x,y,z)
        return fig
    else:
        return go.Figure()

# scatter plot callback

@app.callback(
    Output({'type':'scatter-plot', 'index':MATCH}, 'figure'),
    [Input({'type':'scatter-x-selection', 'index':MATCH}, 'value'),
    Input({'type':'scatter-y-selection', 'index':MATCH}, 'value')],)
def update_graph(x,y):

    if x and y:
        fig = scatter_fig(df,x,y)
        return fig
    else:
        return go.Figure()

# histogram callback

@app.callback(
    Output({'type':'histogram', 'index':MATCH}, 'figure'),
    [Input({'type':'histogram-x-selection', 'index':MATCH}, 'value'),
    Input({'type':'histogram-y-selection', 'index':MATCH}, 'value')])
def update_graph(x,y):

    if x and y:
        fig = histogram_fig(df,x,y)
        return fig
    else:
        return go.Figure()

# toggle on/off

@app.callback(
    Output('interval-component', 'disabled'),
    Output('my-toggle-switch-output', 'children'),
    Input('my-toggle-switch', 'value'),
)
def toggle_interval(on):
    if on:
        return False, "Auto refresh is ON."
    else:
        return True, "Auto refresh is OFF."
    
# refresh button callback

@app.callback(
    [Output({'type':'run-selection', 'index':MATCH}, 'options'),
    Output({'type':'run-selection', 'index':MATCH}, 'value')],
    Input('interval-component', 'n_intervals'),
    prevent_initial_call=True
)
def update_metrics(n_intervals):

        df, byte_columns = fetch_data()
        run_options = [{'label': number, 'value': number} for number in df.index.values]
        selected_run = run_options[0]['value']

        return run_options, selected_run


# new chart callback

@app.callback(
    Output('draggable', 'children'),
    Output('chart-type-dropdown', 'value'),
    State('chart-type-dropdown', 'value'),
    Input('add-chart-btn', 'n_clicks'),
    State('draggable', 'children')
)
def add_chart(chart_type, n_clicks, existing_children):
    if chart_type is None:
        return existing_children, None

    new_chart = create_chart_element(type=chart_type, n_clicks=n_clicks)
    existing_children.append(new_chart)
    
    return existing_children, None
    
# save pos callback

@app.callback(
    Output('dummy-out', 'children'),
    Input('save-button', 'n_clicks'),
    [State({'type':'3d-x-selection', 'index':ALL}, 'value'),
    State({'type':'3d-y-selection', 'index':ALL}, 'value'),
    State({'type':'3d-z-selection', 'index':ALL}, 'value'),
    State({'type':'3d-plot', 'index':ALL}, 'id'),
    State({'type':'scatter-x-selection', 'index':ALL}, 'value'),
    State({'type':'scatter-y-selection', 'index':ALL}, 'value'),
    State({'type':'scatter-plot', 'index':ALL}, 'id'),
    State({'type':'histogram-x-selection', 'index':ALL}, 'value'),
    State({'type':'histogram-y-selection', 'index':ALL}, 'value'),
    State({'type':'histogram', 'index':ALL}, 'id'),
    State({'type':'run-selection', 'index':ALL}, 'value'),
    State({'type':'image-selection', 'index':ALL}, 'value'),
    State({'type':'image-plot', 'index':ALL}, 'id'),
    State('draggable', 'layouts'),
    State('dashboard-name-dropdown', 'value')],
    prevent_initial_call=True
)
def save_position(n_clicks,
                   x_3d, y_3d, z_3d, p_3d,
                  x_sc, y_sc, p_sc,
                  x_his, y_his, p_his,
                  run_img, img_img, p_img,
                   layout, dash_name):
    
    if dash_name:

        all_elements=[]
        all_data=[]

        try:
            with open('assets/data.json', 'r') as file:
                all_data = json.load(file)
        except FileNotFoundError:
            # If the file doesn't exist, continue with an empty list
            pass

        for number, id in enumerate(p_3d):
            type=id['type']
            index=id['index']
            x_val=x_3d[number]
            y_val=y_3d[number]
            z_val=z_3d[number]
            dic={'type':type, 'index':index, 'x_val':x_val, 'y_val':y_val, 'z_val':z_val}
            all_elements.append(dic)

        for number, id in enumerate(p_sc):
            type=id['type']
            index=id['index']
            x_val=x_sc[number]
            y_val=y_sc[number]
            dic={'type':type, 'index':index, 'x_val':x_val, 'y_val':y_val}
            all_elements.append(dic)

        for number, id in enumerate(p_his):
            type=id['type']
            index=id['index']
            x_val=x_his[number]
            y_val=y_his[number]
            dic={'type':type, 'index':index, 'x_val':x_val, 'y_val':y_val}
            all_elements.append(dic)

        for number, id in enumerate(p_img):
            type=id['type']
            index=id['index']
            run=run_img[number]
            img=img_img[number]
            dic={'type':type, 'index':index, 'run':run, 'img':img}
            all_elements.append(dic)

        if all_elements==[]:
            return None
        
        pos = layout
        all_elements = sorted(all_elements, key=lambda x: x['index'])
        found_dash=False
        for dash in all_data:
            if dash['name'] == dash_name:
                dash['elements']=all_elements
                dash['pos']=pos
                found_dash=True

        if not found_dash:
            data={'name': dash_name, 'elements':all_elements, 'pos':pos}
            all_data.append(data)
            
        with open('assets/data.json', 'w') as file:
            json.dump(all_data, file)

    return None
    
# load pos callback

@app.callback(
    [Output('draggable', 'children', allow_duplicate=True),
    Output('draggable', 'layouts'),
    Output('add-chart-btn', 'n_clicks')],
    Input('load-button', 'n_clicks'),
    State('dashboard-name-dropdown', 'value'),
    prevent_initial_call=True
)
def load_position(n_clicks, dash_name):

    if dash_name:
        with open('assets/data.json', 'r') as file:
            loaded_data = json.load(file)

        loaded_data = next((item for item in loaded_data if item["name"] == dash_name), None)

        if loaded_data is None:
            return [], {}, 0
        
        children=[]
        n_charts=0
        for element in loaded_data['elements']:
            type=element['type']
            index=element['index']
            if type!='image-plot':
                x_val=element['x_val']
                y_val=element['y_val']
                if type=='3d-plot':
                    z_val=element['z_val']
                else:
                    z_val=None
                new_chart=create_chart_element(type=type, n_clicks=index, x_val=x_val, y_val=y_val, z_val=z_val)
            else:
                run=element['run']
                img=element['img']
                new_chart=create_chart_element(type=type, n_clicks=index, run=run, image=img)
            children.append(new_chart)
            if index>=n_charts:
                n_charts=index

        layout=loaded_data['pos']

        return children, layout, n_charts
    
    else:
        return [], {}, 0

# create user callback

@callback(
    [Output("dashboard-name-dropdown", "options"),
    Output("dashboard-name-dropdown", "value"),
    Output('new-dashboard', 'value')],
    Input("new-dash-button", "n_clicks"),
    State('new-dashboard', 'value')
)
def create_new_dash(n_clicks, dash_name):

    with open('assets/data.json', 'r') as file:
            loaded_data = json.load(file)
    
    names_list = [item["name"] for item in loaded_data]

    options = [{'label': name, 'value': name} for name in names_list]

    if dash_name:
        options.append({'label': dash_name, 'value': dash_name})
    return options, dash_name, None


# destroy graph callback

@callback(
    Output('draggable', 'children', allow_duplicate=True),
    Input({'type':'close-button', 'index':ALL}, 'n_clicks'),
    State('draggable', 'children'),
    prevent_initial_call=True
)

def destroy_graph(n_clicks, children):

    if n_clicks:
        if n_clicks[0]:
            input_id = callback_context.triggered[0]["prop_id"].split(".")[0]
            print(input_id)
            print(n_clicks)
            if "index" in input_id:
                delete_chart = json.loads(input_id)["index"]
                children = [
                    chart
                    for chart in children
                    if "'index': " + str(delete_chart) not in str(chart)
                ]

    return children



if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_hot_reload=False)

