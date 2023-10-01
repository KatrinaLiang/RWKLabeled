from datetime import datetime
import os
import networkx as nx
import glob as gl
import numpy as np
import pandas as pd
from grakel import RandomWalk, RandomWalkLabeled
from grakel.utils import graph_from_networkx


def get_names(paths):
    df_main = pd.DataFrame()
    dot_name = []
    dot_path_aux = []
    for dot in paths:
        name = dot.split(os.sep)[-1]
        name = name.split('.dot')[0]  # .name.split('.dot')[0].replace("_m", "")
        dot_name.append(name)
        dot_path_aux.append(dot)
    df_main['dot_name'] = dot_name
    df_main['dot_path'] = dot_path_aux
    df_main = df_main.sort_values(by=['dot_name']).reset_index(drop=True)

    return df_main


def read_dot_as_nxGraph(names_paths_df):
    paths = names_paths_df['dot_path']
    cfg_final = []
    for i in paths:
        cfg = nx.drawing.nx_pydot.read_dot(i)
        cfg_final.append(cfg)
    # print(len(cfg_final))
    return cfg_final


def create_df(matrixK, dot_names):
    columnsName = []

    for i in range(np.shape(matrixK)[0]):
        name = 'FM_' + str(i + 1)
        columnsName.append(name)

    index = list(dot_names['dot_name'])

    df_rwk = pd.DataFrame(matrixK, columns=columnsName, index=index)

    return df_rwk


def saveFile(df, path, output):
    if output.find('.') != -1:
        output = output.split('.')[0]

    finalPath = os.path.join(path, output + '.csv')
    df.to_csv(finalPath)
    print('\n DONE! File saved in: \n ', finalPath)


def getOp(myName):
    if myName.find('=') == -1:
        # print(myName)
        return 'ERROR'
    else:
        if myName.find('+') != -1:
            return 8  # 'add'
        elif myName.find('-') != -1:
            return 9  # 'sub'
        elif myName.find('&') != -1:
            return 10  # 'and'
        elif myName.find('cmp') != -1:
            return 11  # 'cmp'
        elif myName.find('cmpg') != -1:
            return 12  # 'cmpg'
        elif myName.find('cmpl') != -1:
            return 13  # 'cmpl'
        elif myName.find('/') != -1:
            return 14  # 'div'
        elif myName.find('==') != -1:
            return 15  # 'eql'
        elif myName.find('>=') != -1:
            return 16  # 'geql'
        elif myName.find('>') != -1:
            return 17  # 'gt'
        elif myName.find('<=') != -1:
            return 18  # 'leql'
        elif myName.find('<') != -1:
            return 19  # 'lt'
        elif myName.find('*') != -1:
            return 20  # 'mul'
        elif myName.find('!=') != -1:
            return 21  # 'neqlL'
        elif myName.find('|') != -1:
            return 22  # 'or'
        elif myName.find('%') != -1:
            return 23  # 'rem'
        elif myName.find('<<') != -1:
            return 24  # 'shl'
        elif myName.find('>>') != -1:
            return 25  # 'shr'
        elif myName.find('xor') != -1:
            return 26  # 'xor'
        else:
            return 27  # 'assi'


def get_node_labels(cfg):
    # print(cfg.edges())
    nodes = cfg.nodes()
    node_names = nx.get_node_attributes(cfg, 'label')
    # print(node_names)
    node_labels = {}

    for n in nodes:
        myName = node_names[n]
        if myName.find('if') != -1:
            nodeLabel = 1  # 'If'
        elif myName.find(':=') != -1:
            nodeLabel = 2  # 'eqltostmt'
        elif myName.find('goto') == 0 or myName.find('goto') == 1:
            nodeLabel = 3  # myName.replace('"', '')  # 'goto'
        elif myName.find('exit') != -1:
            nodeLabel = 4  # 'exit'
        elif myName.find('return') != -1:
            nodeLabel = 5  # myName.replace('"', '')  # 'return'
        elif (myName.find('(') != -1 and myName.find(')') != -1 and (
                myName.find('double') != -1 or myName.find('float') != -1 or myName.find('int') != -1)):
            nodeLabel = 6  # 'assi'
        elif myName.find('(') != -1 and myName.find(')') != -1:
            nodeLabel = 7  # 'fcall'
        else:
            nodeLabel = getOp(myName)
        node_labels[n] = nodeLabel
    return node_labels


if __name__ == '__main__':
    lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    dot_input = 'dotFiles/*'
    resultsPath = 'RWKernels'
    if not os.path.exists(resultsPath):
        os.mkdir(resultsPath)

    dot_path = gl.glob(dot_input)
    names_paths_df = get_names(dot_path)
    cfg_main = np.asarray(read_dot_as_nxGraph(names_paths_df), dtype=object)
    for g in cfg_main:
        if g.has_node('\\n'):
            g.remove_node('\\n')

        mapping = get_node_labels(g)
        nx.set_node_attributes(g, mapping, 'label')

    G_train = list(graph_from_networkx(cfg_main, node_labels_tag='label'))

    for lambda_value in lambdas:
        output_csv_name = "rwk_lambda_{}".format(str(lambda_value).replace(".", "-"))
        start_time = datetime.now()

        # rwk_kernel = RandomWalk(lamda=lambda_value, normalize=False, p=10)
        rwk_kernel = RandomWalkLabeled(lamda=lambda_value, normalize=True, p=10)
        # rwk_kernel_1 = RandomWalkLabeled(lamda=lambda_value, normalize=True, method_type='basline', p=10) # Everything was 1.0

        new_rwk = rwk_kernel.fit_transform(G_train)

        df_rwk = create_df(new_rwk, names_paths_df)
        saveFile(df_rwk, resultsPath, output_csv_name)
        end_time = datetime.now()
        print('RWK calculation took: {}'.format(end_time - start_time))
