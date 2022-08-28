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

'''
  _____ ______ _____ _____ 
 / ____|  ____/ ____/ ____|
| (___ | |__ | |   | |     
 \___ \|  __|| |   | |     
 ____) | |   | |___| |____ 
|_____/|_|    \_____\_____|

This project was ported to SFCC (Suwa Futaba Chemsitry Club) from ykchemi.

This program is released under the following conditions: MIT License

'''




import tkinter as tk
import PySimpleGUI as sg
import numpy as np
import pandas as pd
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
    default_text='3',
    disabled=False,
    key='max_length_of_displayed_results'
)

configure_col_reflecting_configure = sg.Button(
    button_text='計算を実行',
    button_color=('#FFFFFF', '#28af9b'),
    key='reflect_configure'
)

configure_col_clear_canvas = sg.Button(
    button_text='クリア',
    button_color=('#FFFFFF', '#28af9b'),
    key='clear_canvas'
)

configure_col_output_file_name_description = sg.Text(
    text='出力するファイル名',
    text_color='#000000',
    background_color='#FFFFFF'
)

configure_col_output_file_name = sg.Input(
    default_text='data.csv',
    disabled=False,
    key='output_file_name'
)

configure_col_write_data_as_csv = sg.Button(
    button_text='結果をCSVとして出力',
    button_color=('FFFFFF', '#28af9b'),
    key='write_as_csv'
)

configure_col = [
    [configure_col_first_word_description, configure_col_first_word],
    [configure_col_added_or_subtracted_word_description, configure_col_added_or_subtracted_word],
    [configure_col_calculating_mode_description, configure_col_calculating_mode],
    [configure_col_length_of_displayed_results_description, configure_col_length_of_displayed_results],
    [configure_col_reflecting_configure],
    [configure_col_clear_canvas],
    [configure_col_output_file_name_description, configure_col_output_file_name],
    [configure_col_write_data_as_csv]
]

graph_col_description = sg.Text(
    text='言葉の関係をグラフ化',
    text_color='#000000',
    background_color='#FFFFFF'
)

canvas_size_x = 900
canvas_size_y = 700

graph_col_canvas = sg.Canvas(
    size=(canvas_size_x, canvas_size_y),
    background_color="#FFFFFF",
    key='canvas'
)

graph_col = [
    [graph_col_canvas],
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


configure_col_reflecting_configure.bind('<Button>', '_click')
configure_col_clear_canvas.bind('<Button>', '_click')
configure_col_write_data_as_csv.bind('<Button>', '_click')



new_model_path = 'model.pkl'

relay_color = '#e8e3e3'

with open(new_model_path, 'rb') as f:
    model = pickle.load(f)



def makeNerworks(pre_net, node, results_length, mode, calculated_word):
    #nodeの値はnodesに入るようにしといてね（はーと）
    nodes = pre_net.nodes()

    if len(nodes) == 0:
        pre_net.add_node(node, color=relay_color)

    if mode == mode_adding:
        w = '+' + calculated_word + '_' + node
    elif mode == mode_subtracting:
        w = '-' + calculated_word + '_' + node

    pre_net.add_node(w, color='#b9ebe3')
    pre_net.add_edge(node, w)

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
    results_values = [i[0].replace('#', '') for i in results]
    results_values_set = set(results_values)
    elements_intersection = nodes_set & results_values_set

    intersection_deleted_list = [i for i in results_values_set if i not in elements_intersection]

    pre_net.add_nodes_from(intersection_deleted_list, color=relay_color)

    print(pre_net.nodes.data())
    nc = [pre_net.nodes[node]['color'] for node in pre_net.nodes()]

    
    
    edge_list = [[w, i] for i in results_values]

    pre_net.add_edges_from(edge_list)

    pos = nx.spring_layout(pre_net)

    plt.cla()

    nx.draw_networkx(pre_net, font_family='Yu Gothic', labels={node: node for node in pre_net.nodes()}, node_color=nc)

    return pre_net


def write_data_as_csv(graph_data):
    labels = ['初期ワード', '加減モード', '加減ワード', '計算結果']
    nodes = list(graph_data.nodes(data='color', default='#FFFFFF'))
    edges = list(graph_data.edges())
    detect_green = [1 if i[2] == relay_color else 0 for i in nodes]
    calculated_word_and_connected_word = {i:j for i, j in zip()}






#figure kanren
fig = plt.figure(figsize=(canvas_size_x / 100, canvas_size_y / 100))
ax = fig.add_subplot(111)
canvas = graph_col_canvas.TKCanvas

figure_cv = tkagg.FigureCanvasTkAgg(fig, canvas)
figure_cv.draw()
figure_cv.get_tk_widget().pack()




def main():
    G = nx.DiGraph()


    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break

        elif event == 'reflect_configure_click':
            G = makeNerworks(
                G,
                values['first_word'],
                int(values['max_length_of_displayed_results']),
                values['calculating_mode'],
                values['added_or_subtracted_word']
            )
            figure_cv.draw()

        elif event == 'clear_canvas_click':
            plt.cla()
            figure_cv.draw()
            G = nx.DiGraph()

        elif event == 'write_as_csv':
            write_data_as_csv()




    window.close()

if __name__ == '__main__':
    main()