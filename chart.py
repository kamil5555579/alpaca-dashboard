import plotly.graph_objects as go
from dash import html, dcc
import plotly.express as px

# function that creates a component/element that is chart with dropdowns and button

def create_chart_element(type, n_clicks, x_val=None, y_val=None, z_val=None, run=None, image=None,
                          numerical_variable_options=None, histogram_options=None, run_options=None, image_options=None):
    if type=='3d-plot': 
        new_element = html.Div(
            children=[
                html.Button('X', id={'type':'close-button',
                                 'index':n_clicks}, n_clicks=0, className='close-button'),
                dcc.Dropdown(id={'type':'3d-x-selection',
                                 'index':n_clicks}, options=numerical_variable_options, value=x_val, className='dropdown'),
                dcc.Dropdown(id={'type':'3d-y-selection',
                                 'index':n_clicks}, options=numerical_variable_options, value=y_val, className='dropdown'),
                dcc.Dropdown(id={'type':'3d-z-selection',
                                 'index':n_clicks}, options=numerical_variable_options, value=z_val, className='dropdown'),
                dcc.Graph(id={'type':'3d-plot',
                                 'index':n_clicks}, className='graph', figure=go.Figure())
            ],
            className='graph-div'
        )
    if type=='histogram': 
        new_element = html.Div(
            children=[
                html.Button('X', id={'type':'close-button',
                                 'index':n_clicks}, n_clicks=0, className='close-button'),
                dcc.Dropdown(id={'type':'histogram-x-selection',
                                 'index':n_clicks}, options=histogram_options, value=x_val, className='dropdown'),
                dcc.Dropdown(id={'type':'histogram-run-selection',
                                 'index':n_clicks}, options=run_options, value=run, className='dropdown'),
                dcc.Graph(id={'type':'histogram',
                                 'index':n_clicks}, className='graph', figure=go.Figure())
            ],
            className='graph-div'
        )
    if type=='image-plot': 
        new_element = html.Div(
            children=[
                html.Button('X', id={'type':'close-button',
                                 'index':n_clicks}, n_clicks=0, className='close-button'),
                dcc.Dropdown(id={'type':'run-selection',
                                 'index':n_clicks}, options=run_options, value=run, className='dropdown'),
                dcc.Dropdown(id={'type':'image-selection',
                                 'index':n_clicks}, options=image_options, value=image, className='dropdown'),
                dcc.Graph(id={'type':'image-plot',
                                 'index':n_clicks}, className='graph', figure=px.imshow([[0]], color_continuous_scale='gray'))
            ],
            className='graph-div'
        )
    if type=='scatter-plot': 
        new_element = html.Div(
            children=[
                html.Button('X', id={'type':'close-button',
                                 'index':n_clicks}, n_clicks=0, className='close-button'),
                dcc.Dropdown(id={'type':'scatter-x-selection',
                                 'index':n_clicks}, options=numerical_variable_options, value=x_val, className='dropdown'),
                dcc.Dropdown(id={'type':'scatter-y-selection',
                                 'index':n_clicks}, options=numerical_variable_options, value=y_val, className='dropdown'),
                dcc.Graph(id={'type':'scatter-plot',
                                 'index':n_clicks}, className='graph', figure=go.Figure())
            ],
            className='graph-div'
        )
    return new_element

def create_chart_options(type, x_val=None, y_val=None, z_val=None, image=None,
                          numerical_variable_options=None, histogram_options=None, image_options=None):
    """if type=='3d-plot': 
        new_element = html.Div(
            children=[
                dcc.Dropdown(id='3d-x-selection', options=numerical_variable_options, value=x_val, className='dropdown'),
                dcc.Dropdown(id='3d-y-selection', options=numerical_variable_options, value=y_val, className='dropdown'),
                dcc.Dropdown(id='3d-z-selection', options=numerical_variable_options, value=z_val, className='dropdown')
            ],
            className='graph-div'
        )"""
    if type=='histogram': 
        new_element = html.Div(
            children=[
                dcc.Dropdown(id='x-selection', options=histogram_options, value=x_val, className='dropdown'),
                dcc.Dropdown(id='y-selection', options=numerical_variable_options, value=y_val, className='dropdown', style={'display':'none'})
            ],
            className='options-div'
        )
    if type=='image-plot': 
        new_element = html.Div(
            children=[
                dcc.Dropdown(id='x-selection', options=image_options, value=image, className='dropdown'),
                dcc.Dropdown(id='y-selection', options=numerical_variable_options, value=y_val, className='dropdown', style={'display':'none'})
            ],
            className='options-div'
        )
    if type=='scatter-plot': 
        new_element = html.Div(
            children=[
                dcc.Dropdown(id='x-selection', options=numerical_variable_options, value=x_val, className='dropdown'),
                dcc.Dropdown(id='y-selection', options=numerical_variable_options, value=y_val, className='dropdown')
            ],
            className='options-div'
        )
    return new_element
