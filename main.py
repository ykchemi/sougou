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
import PySimpleGUI as sg
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
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

configure_col = [
    [configure_col_first_word_description, configure_col_first_word],
    [configure_col_added_or_subtracted_word_description, configure_col_added_or_subtracted_word]
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


def main():
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break


    window.close()

if __name__ == '__main__':
    main()