from dash import Dash, html, dash_table, dcc, callback, Output, Input, State, MATCH, ALL, Patch, ctx, callback_context
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import dash_draggable
import json
import dash_daq as daq
from plot import scatter_fig, histogram_fig, surface_fig
from database import fetch_data
from chart import create_chart_element

app = Dash(__name__)

# get all the data and types of columns
df, numerical_columns, one_dimensional_columns, two_dimensional_columns = fetch_data()

# dropdowns options
run_options = [{'label': number, 'value': number} for number in df.index.values]
image_options = [{'label': col, 'value': col} for col in two_dimensional_columns]
histogram_options = [{'label': col, 'value': col} for col in one_dimensional_columns]
numerical_variable_options = [{'label': col, 'value': col} for col in numerical_columns]
chart_types = ['3d-plot', 'image-plot', 'histogram', 'scatter-plot']

############################# 
# main layout of the app
#############################

app.layout = html.Div([
    html.Div(children=[
        html.Img(src='assets/alpaca_logo.png'),
        html.Div(id='mid-header', children=[
            html.H1(id='header-title', children=['Alpaca',html.Br(),'Dashboard']), html.Br(),
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
                                            html.Button('save', id='new-dash-button', n_clicks=0, type='button')], className='row')
        ]),
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


############################# 
# all the callbacks
#############################

# image plot callback

@app.callback(
    Output({'type':'image-plot', 'index':MATCH}, 'figure'),
    [Input({'type':'run-selection', 'index':MATCH}, 'value'),
    Input({'type':'image-selection', 'index':MATCH}, 'value')])
def update_graph(selected_run, selected_img):

    if selected_run and selected_img:
        data = df[selected_img][selected_run]
        if data is not None and data.any():
            img = px.imshow(df[selected_img][selected_run], title=two_dimensional_columns[0], zmin=0, zmax=np.max(df[selected_img][selected_run]) / 3,  #
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
    Input({'type':'histogram-run-selection', 'index':MATCH}, 'value')])
def update_graph(x,run):

    if x and run:
        fig = histogram_fig(df,x,run)
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

        df, numerical_columns, one_dimensional_columns, two_dimensional_columns = fetch_data()
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

    new_chart = create_chart_element(type=chart_type, n_clicks=n_clicks, numerical_variable_options=numerical_variable_options,
                                     image_options=image_options, run_options=run_options, histogram_options=histogram_options)
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
    State({'type':'histogram-run-selection', 'index':ALL}, 'value'),
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
                  x_his, run_his, p_his,
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
            run=run_his[number]
            dic={'type':type, 'index':index, 'x_val':x_val, 'run':run}
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
            if type=='histogram':
                run=element['run']
                x_val=element['x_val']
                new_chart=create_chart_element(type=type, n_clicks=index, run=run, x_val=x_val, histogram_options=histogram_options, run_options=run_options)
            elif type=='image-plot':
                run=element['run']
                img=element['img']
                new_chart=create_chart_element(type=type, n_clicks=index, run=run, image=img, run_options=run_options, image_options=image_options)
            else:
                x_val=element['x_val']
                y_val=element['y_val']
                if type=='3d-plot':
                    z_val=element['z_val']
                else:
                    z_val=None
                new_chart=create_chart_element(type=type, n_clicks=index, x_val=x_val, y_val=y_val, z_val=z_val, numerical_variable_options=numerical_variable_options)

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
    Output('draggable', 'layouts', allow_duplicate=True),
    Output('draggable', 'children', allow_duplicate=True),
    Input({'type':'close-button', 'index':ALL}, 'n_clicks'),
    State('draggable', 'children'),
    State('draggable', 'layouts'),
    prevent_initial_call=True
)

def destroy_graph(n_clicks, children, layout):

    position = next((index for index, num in enumerate(n_clicks) if num != 0), None)
    if position is not None:

        input_id = callback_context.triggered[0]["prop_id"].split(".")[0]

        if 0 <= position < len(layout["lg"]):
            layout["lg"] = layout["lg"][:position] + layout["lg"][position + 1:]
            for idx, item in enumerate(layout["lg"]):
                item["i"] = str(idx)

        if "index" in input_id:
            delete_chart = json.loads(input_id)["index"]
            children = [
                    chart
                    for chart in children
                    if "'index': " + str(delete_chart) not in str(chart)
                ]
        
    return layout, children



if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_hot_reload=False)

