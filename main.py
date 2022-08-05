'''


 _   _                      _   _                   
| \ | |                    | \ | |                  
|  \| |_   _  __ _ _ __    |  \| |_   _  __ _ _ __  
| . ` | | | |/ _` | '_ \   | . ` | | | |/ _` | '_ \ 
| |\  | |_| | (_| | | | |  | |\  | |_| | (_| | | | |
|_| \_|\__, |\__,_|_| |_|  |_| \_|\__, |\__,_|_| |_|
        __/ |                      __/ |            
       |___/                      |___/             
                                                   

'''




import tkinter as tk
from turtle import back
import PySimpleGUI as sg
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import japanize_matplotlib
import matplotlib.backends.backend_tkagg as tkagg

import pickle
from MeCab import Model
from gensim.models import KeyedVectors




configure_col_first_word_description = sg.Text(
    text='初期ワード',
    text_color='#000000',
    background_color='#FFFFFF'
)

configure_col_first_word = sg.Input(
    default_text='トイレ',
    disabled=False,
    key='first_word'
)

configure_col_added_or_subtracted_word_description = sg.Text(
    text='初期ワードから加減するワード',
    text_color='#000000',
    background_color='#FFFFFF'
)

configure_col_added_or_subtracted_word = sg.Input(
    default_text='女神',
    disabled=False,
    key='added_or_subtracted_word'
)

configure_col_calculating_mode_description = sg.Text(
    text='加減モード変換',
    text_color='#000000',
    background_color='#FFFFFF'
)

mode_adding = '+'
mode_subtracting = '-'
configure_col_calculating_mode = sg.Combo(
    [mode_adding, mode_subtracting],
    default_value=mode_adding,
    readonly=True,
    background_color='white',
    text_color='#000000',
    size=(50, 1),
    enable_events=True,
    key='calculating_mode'
)

configure_col_length_of_displayed_results_description = sg.Text(
    text='最大追加数',
    text_color='#000000',
    background_color='#FFFFFF'
)

configure_col_length_of_displayed_results = sg.Input(
    default_text='10',
    disabled=False,
    key='max_length_of_displayed_results'
)

configure_col_reflecting_configure = sg.Button(
    button_text='計算を実行',
    button_color=('#FFFFFF', '#28af9b'),
    key='reflect_configure'
)

configure_col = [
    [configure_col_first_word_description, configure_col_first_word],
    [configure_col_added_or_subtracted_word_description, configure_col_added_or_subtracted_word],
    [configure_col_calculating_mode_description, configure_col_calculating_mode]
]



graph_col_description = sg.Text(
    text='言葉の関係をグラフ化',
    text_color='#000000',
    background_color='#FFFFFF'
)

canvas_size_x = 500
canvas_size_y = 500

graph_col = sg.Canvas(
    size=(canvas_size_x, canvas_size_y),
    background_color="#FFFFFF",
    key='canvas'
)

graph_col = [
    [graph_col],
    [graph_col_description]
]


sg.theme('DefaultNoMoreNagging')
sg.theme_background_color('#FFFFFF')


layout = [
    [
        sg.Column(configure_col, element_justification='right'),
        sg.Column(graph_col, element_justification='right')
    ]
]

window = sg.Window(
    title='Genbu',
    layout=layout
)

window.finalize()


new_model_path = 'model.pkl'

with open(new_model_path, 'rb') as f:
    model = pickle.load(f)



def makeNerworks(pre_net, node, results_length, mode, calculated_word):
    #nodeの値はnodesに入るようにしといてね（はーと）
    nodes = pre_net.nodes()

    if len(nodes) == 0:
        pre_net.add_node(node)

    if mode == mode_adding:
        results = model.most_similar(
            positive=[node, calculated_word],
            topn=results_length
        )
    elif mode == mode_subtracting:
        results = model.most_similar(
            positive=[node],
            negative=[calculated_word],
            topn=results_length
        )
    else:
        results = None

    nodes_set = set(nodes)
    results_values = [i[0] for i in results]
    results_values_set = set(results_values)
    elements_intersection = nodes_set & results_values_set

    intersection_deleted_list = [i for i in results_values_set if i not in elements_intersection]

    pre_net.add_nodes_from(intersection_deleted_list)

    edge_list = [(node, i) for i in results_values]

    pre_net.add_edges_from(edge_list)

    return pre_net







#figure kanren
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)




def main():
    G = nx.DiGraph()


    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break


    window.close()

if __name__ == '__main__':
    main()