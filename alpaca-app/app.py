from dash import Dash, html, dcc, callback, Output, Input, State, MATCH, ALL, callback_context
import numpy as np
import dash_draggable
import json
import dash_daq as daq
import dash_bootstrap_components as dbc
from database import initial_fetch_data, fetch_run, get_column_type, execute_alpaca_script, wait_for_database
from dashboards import save_or_update_dashboard, dashboard_name_exists, get_dashboard_by_name, get_all_dashboard_names
from chart import create_chart_options, create_chart, serialize_df, deserialize_df
from tree import create_tree, generate_legend
import time
import dash_auth

# Keep this out of source code repository - save in a file or a database
VALID_USERNAME_PASSWORD_PAIRS = {
    'hello': 'world'
}

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)
server = app.server

# initial data from alpaca
execute_alpaca_script(387116, 387116)

# Wait for database to be ready
if wait_for_database("alpaca"):
    print("Database is ready, executing query...")
    time.sleep(1)
    # get all the data and types of columns
    runs, columns_dic = initial_fetch_data()
else:
    print("Database did not become ready within the specified time.")

# dropdowns options
run_options = [{'label': number, 'value': number} for number in runs]
image_options = [{'label': col, 'value': col} for col in columns_dic['two_dimensional_columns']]
one_dimensional_options = [{'label': col, 'value': col} for col in columns_dic['one_dimensional_columns']]
numerical_options = [{'label': col, 'value': col} for col in columns_dic['numerical_columns']]
chart_types = ['image-plot', 'histogram', 'scatter-plot', 'number']
all_columns=np.concatenate((columns_dic['numerical_columns'] ,columns_dic['one_dimensional_columns'], columns_dic['two_dimensional_columns']))


# modals (for creating user and creating chart)

modals = html.Div([dbc.Modal(id={'type': 'modal', 'index':0}, children=
        [
            dbc.ModalHeader(id='errors', children=["To jest nagłówek modalu"]),
            dbc.ModalBody(children=[
                html.Div('Title:'), dbc.Input(id='chart-title', placeholder='Name your element', type='text', class_name='dropdown'),
                html.Div([html.Div('Type:'), dcc.Dropdown(id='chart-type-dropdown', 
                                        options=[{'label': chart_type, 'value': chart_type} for chart_type in chart_types],
                                        placeholder='Select Chart Type', className='dropdown')
                                        ]),
                html.Div(id='chart-options', children=[html.Div(
            children=[
                dcc.Dropdown(id='x-selection', options=one_dimensional_options, className='dropdown', style={'display':'none'}),
                dcc.Dropdown(id='y-selection', options=numerical_options, className='dropdown', style={'display':'none'})
            ],
            className='options-div'
        )])
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
            dbc.ModalHeader("Create new dashboard"),
            dbc.ModalBody(children=[
                dbc.Input(id='dashboard-title', 
                                    type='text',
                                    placeholder='Name your dashboard')
            ]),
            dbc.ModalFooter([
                dbc.Button("Close", id={'type': 'close-modal-btn', 'index':1}, className="button"),
                 dbc.Button('Save', id='new-dash-button', n_clicks=0, color="primary", className="button"),
        ]),
        ],
        centered=True,
    )])

# mid header

mid_header = html.Div(id='mid-header', children=[
            html.H1(id='header-title', children=['Alpaca Dashboard(s)']),
            html.Div(className='row', children=[
                    dcc.Dropdown(id='dashboard-name-dropdown',
                        placeholder='Select your dashboard', className='dropdown'),
                    dbc.Button('Load', id='load-button', n_clicks=0, color="primary", className="button"),
                    dbc.Button('Save', id='save-button', n_clicks=0, color="primary", class_name='button'),
                    dbc.Button("Create new dash", id={'type': 'open-modal-btn', 'index':1}, color="primary", className="button")
                    ])
        ])


# right header

right_header = html.Div(children=[
            html.Div(className='column', style={'text-align': 'center', 'margin-right': '17px'}, children=[
                html.Img(src='assets/AEgIS-logo.png', id='aegis-logo'),
                html.Div(className='row', children=[
                dcc.Loading(style={'float':'left'}, children=[
                    'Run:',
                    dcc.Dropdown(id='run-selection', options=run_options,
                        value=run_options[0]['value'], clearable=False,
                        style={'width':'170px', 'margin-bottom': '5px', 'font-size': 'large', 'float':'right'})
                ])
                        ]),
                daq.ToggleSwitch(id='my-toggle-switch', value=False),
                html.Div(id='my-toggle-switch-output')
                    ]),
                html.Div(className='column', children=[
                html.Div('Specify the runs:'),
                dcc.Checklist(id='from-alpaca', options=[{'label':'From Alpaca', 'value': True}], style={'font-size':'small'}),
                dbc.Input(type='number', placeholder='first run', class_name='dropdown', id='first-run'),
                dbc.Input(type='number', placeholder='last run', class_name='dropdown', id='last-run'),
                dbc.Button('Search', id='search-button', n_clicks=0, color="primary", className="button"),
                dcc.Interval(id='interval-component', interval=5*1000, n_intervals=0)
                    ]),
                ],
            id='right-header')

############################# 
# main layout of the app
#############################

app.layout = html.Div([
    html.Div(children=[
        html.Img(src='assets/alpaca_logo.png'),
        mid_header,
        right_header,
        ], id='header-area'),
       html.Div(className='row', children=[
            html.Div(id='draggable-area', children=[dash_draggable.ResponsiveGridLayout(id='draggable', children=[])]),
            html.Div(id='right-panel', children=[
                html.Div('Choose an observable and press'),
                dbc.Button("Create", id={'type': 'open-modal-btn', 'index':0}, color="primary", style={'margin': '10px 10px 10px 0px'}),
                generate_legend(),
                create_tree(all_columns, columns_dic=columns_dic)])
            ]), dcc.Store(id='created-graphs', data=[]), dcc.Store(id='dataframe'), modals
            ])


############################# 
# all the callbacks
#############################

# updating chart options after choosing chart type

'''@app.callback(
    Output('chart-options', 'children'),
    Input('chart-type-dropdown', 'value')
)
def update_options(chart_type):
    if chart_type is None:
        return html.Div(
            children=[
                dcc.Dropdown(id='x-selection', options=one_dimensional_options, className='dropdown', style={'display':'none'}),
                dcc.Dropdown(id='y-selection', options=numerical_options, className='dropdown', style={'display':'none'})
            ],
            className='options-div'
        ) # dummy options because there are errors if x and y selection dont exist

    new_options = create_chart_options(type=chart_type, numerical_options=numerical_options,
                                     image_options=image_options, one_dimensional_options=one_dimensional_options)
    
    return new_options'''

# creating new chart

@app.callback(
    Output('draggable', 'children'),
    Output('chart-type-dropdown', 'value'),
    Output('created-graphs', 'data'),
    Output('chart-title', 'value'),
    Output({'type': 'modal', 'index':0}, "is_open", allow_duplicate=True),
    State('chart-type-dropdown', 'value'),
    Input('add-chart-btn', 'n_clicks'),
    State({'type': 'modal', 'index':0}, "is_open"),
    State('draggable', 'children'),
    State('created-graphs', 'data'),
    State('x-selection', 'value'),
    State('y-selection', 'value'),
    State('chart-title', 'value'),
    State('dataframe', 'data'),
    prevent_initial_call=True
)
def add_chart(chart_type, n_clicks, is_open, existing_children, existing_data, selected_x, selected_y, title, df):

    if n_clicks and is_open:
        df=deserialize_df(df)
        is_open=False
        if chart_type is None:
            return existing_children, None, existing_data, None, is_open

        new_element, chart_dic = create_chart(chart_type, title, df, n_clicks, selected_x, selected_y)
        existing_data.append(chart_dic)
        existing_children.append(new_element)
    
    return existing_children, None, existing_data, None, is_open
    
# saving all graphs and their positions to json

@app.callback(
    Output('load-button', 'n_clicks'),
    Input('save-button', 'n_clicks'),
    [State('draggable', 'layouts'),
    State('dashboard-name-dropdown', 'value'),
    State('created-graphs', 'data'),],
    prevent_initial_call=True
)
def save_position(n_clicks, layout, dash_name, created_graphs):
    
    if dash_name and created_graphs:
        
        save_or_update_dashboard(dashboard_name=dash_name, elements=created_graphs, layout=layout)

    return 1
    
# loading graphs and positions from json

@app.callback(
    [Output('draggable', 'children', allow_duplicate=True),
    Output('draggable', 'layouts'),
    Output('add-chart-btn', 'n_clicks'),
    Output('created-graphs', 'data', allow_duplicate=True),],
    Input('load-button', 'n_clicks'),
    State('dashboard-name-dropdown', 'value'),
    State('dataframe', 'data'),
    prevent_initial_call=True
)
def load_position(n_clicks, dash_name, df):

    if dash_name:
        
        df=deserialize_df(df)

        if not dashboard_name_exists(dash_name):
            return [], {}, 0, []
        
        children=[]
        created_graphs=[]
        n_charts=0

        elements, layout = get_dashboard_by_name(dashboard_name=dash_name)

        for element in elements:
            type=element['type']
            index=element['index']
            title=element['title']
            x_val=element['x_val']
            y_val=None
            if 'y_val' in element.keys():
                y_val=element['y_val']

            new_chart, chart_dic = create_chart(type, title, df, index, x_val, y_val)

            children.append(new_chart)
            created_graphs.append(chart_dic)
            if index>=n_charts:
                n_charts=index

        return children, layout, n_charts, created_graphs
    
    else:
        return [], {}, 0, []

# creating new dashboard

@callback(
    [Output("dashboard-name-dropdown", "options"),
    Output("dashboard-name-dropdown", "value"),
    Output('dashboard-title', 'value')],
    Input("new-dash-button", "n_clicks"),
    State('dashboard-title', 'value')
)
def create_new_dash(n_clicks, dash_name):

    names_list = get_all_dashboard_names()

    options = [{'label': name, 'value': name} for name in names_list]

    if dash_name:
        options.append({'label': dash_name, 'value': dash_name})
    return options, dash_name, None


# deleting graphs

@callback(
    Output('draggable', 'layouts', allow_duplicate=True),
    Output('draggable', 'children', allow_duplicate=True),
    Output('created-graphs', 'data', allow_duplicate=True),
    Input({'type':'close-button', 'index':ALL}, 'n_clicks'),
    State('draggable', 'children'),
    State('draggable', 'layouts'),
    State('created-graphs', 'data'),
    prevent_initial_call=True
)

def destroy_graph(n_clicks, children, layout, created_graphs):

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
            created_graphs = [graph for graph in created_graphs if graph['index']!=delete_chart]
        
    return layout, children, created_graphs

# auto-refreshing run number to most recent

@app.callback(
    [Output('run-selection', 'options'),
    Output('run-selection', 'value')],
    Input('interval-component', 'n_intervals'),
    prevent_initial_call='initial_duplicate'
)
def update_metrics(n_intervals):

    runs, columns_dic = initial_fetch_data()
    run_options = [{'label': number, 'value': number} for number in runs]
    selected_run = run_options[0]['value']

    return run_options, selected_run

# run-selection callback

@app.callback(
    Output('dataframe', 'data'),
    Input('run-selection', 'value')
)
def update_df_from_run(selected_run):

    df=fetch_run(selected_run)
    data = serialize_df(df)

    return data

# when run is changed, save and load again with new run number

@app.callback(
    Output('save-button', 'n_clicks'),
    Input('run-selection', 'value')
)
def update_with_run(selected_run):

    return 1

# auto-refresh toggle on/off

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

# get observable from checklist to dropdown that creates chart

@app.callback(
    Output('errors', 'children'),
    Output('chart-type-dropdown', 'value', allow_duplicate=True),
    Output('chart-options', 'children', allow_duplicate=True),
    Input({'type': 'open-modal-btn', 'index':0}, "n_clicks"),
    State({'type': 'checklist', 'index':ALL}, 'value'),
    prevent_initial_call=True
)
def get_observable(n_clicks, values):
    if n_clicks:

        flat_list = [item for sublist in values if sublist is not None for item in sublist if item is not None]
            
        options = html.Div(
                children=[
                    dcc.Dropdown(id='x-selection', options=one_dimensional_options, className='dropdown', style={'display':'none'}),
                    dcc.Dropdown(id='y-selection', options=numerical_options, className='dropdown', style={'display':'none'})
                ],
                className='options-div'
            )
        
        chart_type = None
        error = None
        y_val=None
            
        types = [get_column_type(item, columns_dic) for item in flat_list]

        if len(flat_list)==0:
            error = 'No observable chosen'

        elif len(flat_list)>2:
            error = 'Too many observables'
            
        elif len(flat_list)==2 and types[0]!=types[1]:
            error = 'Different types of observables'

        else:
            if types[0]=="numerical_columns":
                error = 'Create chart'
                chart_type = 'number'
            elif types[0]=="one_dimensional_columns" and len(flat_list)==1:
                error = 'Create chart'
                chart_type = 'histogram'
            elif types[0]=="one_dimensional_columns" and len(flat_list)==2:
                error = 'Create chart'
                chart_type = 'scatter-plot'
                y_val=flat_list[1]
            elif types[0]=="two_dimensional_columns":
                error = 'Create chart'
                chart_type = 'image-plot'
            
            x_val = flat_list[0]

        if chart_type is not None:
            options = create_chart_options(type=chart_type, x_val=x_val, y_val=y_val, numerical_options=numerical_options,
                                        image_options=image_options, one_dimensional_options=one_dimensional_options)

        return error, chart_type, options
    
# search for specified runs

@app.callback(
    Output('run-selection', 'options', allow_duplicate=True),
    Output('run-selection', 'value', allow_duplicate=True),
    Input('search-button', 'n_clicks'),
    State('dataframe', 'data'),
    State('first-run', 'value'),
    State('last-run', 'value'),
    State('from-alpaca', 'value'),
    prevent_initial_call=True
)
def run_tool(n_clicks, df, first_run, last_run, from_alpaca):
    if n_clicks > 0 and first_run is not None and last_run is not None:
        # Prepare the command to invoke the tool with inputs

        if from_alpaca:
            # Execute the command and wait for it to finish
            execute_alpaca_script(first_run, last_run)
            time.sleep(1)

        # Subprocess finished without error, now fetch the data
        runs, columns_dic = initial_fetch_data(first_run=first_run, last_run=last_run)
        run_options = [{'label': number, 'value': number} for number in runs]
        selected_run = run_options[0]['value']
        
        return run_options, selected_run


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8050, debug=True, dev_tools_hot_reload=False)

