from dash import html, dcc
from database import get_column_type

def get_color(column_type):
    if column_type=="numerical_columns":
        return 'purple'
    if column_type=="one_dimensional_columns":
        return 'green'
    if column_type=="two_dimensional_columns":
        return 'blue'
    
data_info = [
    {'name': 'Number', 'color': get_color("numerical_columns")},
    {'name': '1D array', 'color': get_color("one_dimensional_columns")},
    {'name': '2D array', 'color': get_color("two_dimensional_columns")},
]

def generate_legend_item(name, color):
    return html.Div(
        className='legend-item',
        children=[
            html.Div(style={'background-color': color}, className='legend-color'),
            html.Div(name, className='legend-label')
        ]
    )

def generate_legend():
    legend=[generate_legend_item(data['name'], data['color']) for data in data_info]
    return html.Div(legend, className='legend')

def create_tree(items, parent_full_name=None, checklist_index=0, columns_dic=None):
    trees = {}
    words = []

    for item in items:
        parts = item.split(' ')
        first_word = parts[0]
        full_name = first_word if parent_full_name is None else f"{parent_full_name} {first_word}"

        if len(parts) < 2:
            words.append(full_name)
        else:
            rest_of_words = ' '.join(parts[1:])
            if first_word not in trees:
                trees[first_word] = [rest_of_words]
            else:
                trees[first_word].append(rest_of_words)

    if not trees:
        options = [{'label': html.Div([word.split(' ')[-1]], style={'color': get_color(get_column_type(word, columns_dic))}),'value': word} for word in words]
        return html.Div(dcc.Checklist(id={'type':'checklist', 'index':checklist_index}, options=options, labelStyle={"display": "flex", "align-items": "center"},))


    tree_elements = []
    index=1
    for first_word, children in trees.items():
        tree = html.Details([
            html.Summary(first_word)
        ])
        full_name = first_word if parent_full_name is None else f"{parent_full_name} {first_word}"
        tree.children.append(create_tree(children, parent_full_name=full_name, checklist_index=checklist_index+index+10, columns_dic=columns_dic))
        tree_elements.append(tree)
        index+=1

    return html.Div(tree_elements)
