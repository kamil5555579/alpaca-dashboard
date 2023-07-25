from dash import html, dcc

def create_tree(items, parent_full_name=None, checklist_index=0):
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
        return html.Div(dcc.Checklist(id={'type':'checklist', 'index':checklist_index}, options=[{'label': word.split(' ')[-1], 'value': word} for word in words]))


    tree_elements = []
    index=1
    for first_word, children in trees.items():
        tree = html.Details([
            html.Summary(first_word)
        ])
        tree.children.append(create_tree(children, parent_full_name=first_word, checklist_index=checklist_index+index+10))
        tree_elements.append(tree)
        index+=1

    return html.Div(tree_elements)