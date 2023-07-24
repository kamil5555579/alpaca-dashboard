from dash import Dash, html, dash_table, dcc, callback, Output, Input, State, MATCH, ALL, Patch, ctx, callback_context
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import dash_draggable
import json
import dash_daq as daq
import dash_bootstrap_components as dbc
from plot import scatter_fig, histogram_fig, surface_fig
from database import fetch_data
from chart import create_chart_options, create_chart

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# get all the data and types of columns
df, numerical_columns, one_dimensional_columns, two_dimensional_columns = fetch_data()

created_graphs=[]

# dropdowns options
run_options = [{'label': number, 'value': number} for number in df.index.values]
image_options = [{'label': col, 'value': col} for col in two_dimensional_columns]
histogram_options = [{'label': col, 'value': col} for col in one_dimensional_columns]
numerical_variable_options = [{'label': col, 'value': col} for col in numerical_columns]
chart_types = ['3d-plot', 'image-plot', 'histogram', 'scatter-plot']
all_columns=np.concatenate((numerical_columns,one_dimensional_columns, two_dimensional_columns))

def create_tree(items, parent_full_name=None):
    trees = {}
    words = []

    for item in items:
        parts = item.split(' ')
        first_word = parts[0]
        full_name = item if parent_full_name is None else f"{parent_full_name} {item}"

        if len(parts) < 2:
            words.append(full_name)
        else:
            rest_of_words = ' '.join(parts[1:])
            if first_word not in trees:
                trees[first_word] = [rest_of_words]
            else:
                trees[first_word].append(rest_of_words)

    if not trees:
        return html.Div(dcc.Checklist(options=[{'label': word.split(' ')[-1], 'value': word} for word in words]))


    tree_elements = []
    for first_word, children in trees.items():
        tree = html.Details([
            html.Summary(first_word)
        ])
        tree.children.append(create_tree(children, parent_full_name=first_word))
        tree_elements.append(tree)

    return html.Div(tree_elements)

numerical_details=[html.Summary('numerical observables')]
for text in numerical_columns:
    numerical_details.append(html.Div(text))

array_details=[html.Summary('1D array observables')]
for text in one_dimensional_columns:
    array_details.append(html.Div(text))

array_2d_details=[html.Summary('2D array observables')]
for text in two_dimensional_columns:
    array_2d_details.append(html.Div(text))

panel_details=[html.Summary('one run analysis'), html.Details(numerical_details),
                html.Details(array_details), html.Details(array_2d_details)]

############################# 
# main layout of the app
#############################

app.layout = html.Div([
    html.Div(children=[
        html.Img(src='assets/alpaca_logo.png'),
        html.Div(id='mid-header', children=[
            html.H1(id='header-title', children=['Alpaca Dashboard(s)']),
            html.Div(className='row', children=[
                    dcc.Dropdown(id='dashboard-name-dropdown',
                        placeholder='Select your dashboard', className='dropdown'),
                    dbc.Button('Load', id='load-button', n_clicks=0, color="primary", className="button"),
                    dbc.Button('Save', id='save-button', n_clicks=0, color="primary", class_name='button'),
                    dbc.Button("Create new dash", id={'type': 'open-modal-btn', 'index':1}, color="primary", className="button"),
                    dbc.Button("Create new object", id={'type': 'open-modal-btn', 'index':0}, color="primary", className="button"),
                    ])
        ]),
        html.Div(children=[
            html.Img(src='assets/AEgIS-logo.png', id='aegis-logo'), html.Br(),
            'Last run: ', df.index[0],html.Br(),
            daq.ToggleSwitch(id='my-toggle-switch', value=False),
            html.Div(id='my-toggle-switch-output'),
            dcc.Interval(id='interval-component', interval=5*1000, n_intervals=0)],
            id='right-header')
        ], id='header-area'),
       html.Div(className='row', children=[
            html.Div(id='draggable-area', children=[dash_draggable.ResponsiveGridLayout(id='draggable', children=[], nrows=4, ncols=4, clearSavedLayout=True)]),
            html.Div(id='right-panel', children=create_tree(all_columns))]),
        html.Div(id='dummy-out'),
        dbc.Modal(id={'type': 'modal', 'index':0}, children=
        [
            dbc.ModalHeader("To jest nagłówek modalu"),
            dbc.ModalBody(children=[
                html.Div([dcc.Dropdown(id='chart-type-dropdown', 
                                        options=[{'label': chart_type, 'value': chart_type} for chart_type in chart_types],
                                        placeholder='Select Chart Type', className='dropdown')
                                        ]),
                html.Div(id='chart-options')
            ]),
            dbc.ModalFooter([
                dbc.Button("Close", id={'type': 'close-modal-btn', 'index':0}, className="button"),
                dbc.Button('Add Chart', id='add-chart-btn', n_clicks=0, color="primary", className="button")
        ]),
        ],
        centered=True,
        ),
        dbc.Modal(id={'type': 'modal', 'index':1}, children=
        [
            dbc.ModalHeader("To jest nagłówek modalu"),
            dbc.ModalBody(children=[
                dcc.Input(id='new-dashboard', 
                                    type='text',
                                    placeholder='create new dash', style={'width': '200px'})
            ]),
            dbc.ModalFooter([
                dbc.Button("Close", id={'type': 'close-modal-btn', 'index':1}, className="button"),
                 dbc.Button('save', id='new-dash-button', n_clicks=0, color="primary", className="button"),
        ]),
        ],
        centered=True,
    )
            ])


############################# 
# all the callbacks
#############################

# modal open and close

@app.callback(
    Output({'type': 'modal', 'index':MATCH}, 'is_open'),
    [Input({'type': 'open-modal-btn', 'index':MATCH}, "n_clicks"), Input({'type': 'close-modal-btn', 'index':MATCH}, "n_clicks")],
    [State({'type': 'modal', 'index':MATCH}, "is_open")],
)
def toggle_modal(open_clicks, close_clicks, is_open):
    if open_clicks or close_clicks:
        return not is_open
    return is_open

# chart type dropdown

@app.callback(
    Output('chart-options', 'children'),
    Input('chart-type-dropdown', 'value')
)
def update_options(chart_type):
    if chart_type is None:
        return []

    new_options = create_chart_options(type=chart_type, numerical_variable_options=numerical_variable_options,
                                     image_options=image_options, histogram_options=histogram_options)
    
    return new_options


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
    State('draggable', 'children'),
    State('x-selection', 'value'),
    State('y-selection', 'value'),
    prevent_initial_call=True
)
def add_chart(chart_type, n_clicks, existing_children, selected_x, selected_y):

    if n_clicks:
        if chart_type is None:
            return existing_children, None

        new_element, chart_dic = create_chart(chart_type, 'joł', df, n_clicks, selected_x, selected_y, 387122)
        created_graphs.append(chart_dic)
        existing_children.append(new_element)
    
    return existing_children, None
    
# save pos callback

@app.callback(
    Output('dummy-out', 'children'),
    Input('save-button', 'n_clicks'),
    [State('draggable', 'layouts'),
    State('dashboard-name-dropdown', 'value')],
    prevent_initial_call=True
)
def save_position(n_clicks, layout, dash_name):
    
    if dash_name:

        all_elements=created_graphs
        all_data=[]

        try:
            with open('assets/data.json', 'r') as file:
                all_data = json.load(file)
        except FileNotFoundError:
            # If the file doesn't exist, continue with an empty list
            pass

        if all_elements==[]:
            return None
        
        pos = layout

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
                x_val=element['x_val']
                img = px.imshow(df[x_val].iloc[1], title=two_dimensional_columns[0], zmin=0, zmax=np.max(df[x_val].iloc[1]) / 3,  #
                                    color_continuous_scale='GnBu')
                new_chart = html.Div(
            children=[html.Button('x', id={'type':'close-button',
                                 'index':n_clicks}, n_clicks=0, className='close-button'),
                                 dcc.Graph(id={'type':'image-plot',
                                 'index':n_clicks}, className='graph', figure=img)],
            className='graph-div'
        )
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

