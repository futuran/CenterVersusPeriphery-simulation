import numpy as np
from numpy.core.numeric import Inf
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import Point, LineString, Polygon

import torch
import torch.nn as nn

def get_adjacency_matrix(centroid_list):
    """隣接行列をロード"""
    try:
        adjc = np.load('adjacency_matrix.npy')
    except:
        print('There is no npy file so that calucuating now....')
        adjc = np.full((len(centroid_list), len(centroid_list)),Inf)
        for b, base_city in enumerate(centroid_list):
            for h, hyp_city in enumerate(centroid_list):
                adjc[b,h] = base_city.distance(hyp_city)
        np.save('adjacency_matrix.npy',adjc)
    return adjc

def convert_zerone(centroid_list,matrix,threshold=0.25):
    line_list = []  #要素iには、都市iがエッジをもつすべての都市のインデックス
    linestring_list=[] #上記の都市のエッジについてgpdのLINESTRINGオブジェクトを格納
    size = len(matrix)
    for i in range(size):
        tmp = []
        for j in range(size):
            if matrix[i,j] > threshold:
                matrix[i,j] = 0
            else:
                matrix[i,j] = 1
                tmp.append(j)
                linestring_list.append(LineString([centroid_list[i],centroid_list[j]]))
        line_list.append(tmp)
    return matrix, line_list, linestring_list

def generate_init_matrix(city_id, city_size,d=128):
    """言語行列を初期化"""
    div_num = int(city_size/d)+2
    lang_matrix = torch.zeros((city_size,d))   # size : 都市数*言語数(128)
    lang_matrix[:,int(city_id/div_num)+1] = 1
    #print(lang_matrix)
    return lang_matrix



def update_matrix(matrixes,line_list):
    """言語行列を更新"""
    out_matrixes = []

    for index, matrix in enumerate(matrixes):
        tmp = torch.zeros_like(matrix)
        print(tmp.shape)
        for inter_city in line_list[index]:
            tmp = torch.add(tmp,matrixes[inter_city])

        out_matrixes.append(torch.add(0.9*matrix, 0.1*tmp))
    return out_matrixes


def gpd_language(data,lang_matrixs,lang):
    """
    言語に関する行列から「言語」を算出する
    lang_matrix:言語に関する行列（文法行列　or 語彙行列）
        size:市区町村数(1907)*言語の種類数(128)
    lang:各市区町村の言語のリスト
    """
    for index, matrix in enumerate(lang_matrixs):
        new_lang = int(torch.max(torch.sum(matrix,0),0)[1])
        lang[index] = new_lang
    data.language = gpd.GeoDataFrame(lang)
    return lang

def main():
    fp = "./japan_ver83/japan_ver83.shp"
    data =  gpd.read_file(fp)
    data['language'] = 0

    centroid_list = data.geometry.centroid.to_list()    # 各市区町村の中心のリスト
    adjc = get_adjacency_matrix(centroid_list)          # 隣接行列のリスト
    adjc, line_list, linestring_list = convert_zerone(centroid_list, adjc)  # 0-1に変換

    city_size = len(adjc)
    print("| 市区町村数(city_size):{}".format(city_size))

    # 隣接としへの線分をグラフで表示
    #graph=gpd.GeoDataFrame({'geometry':linestring_list})
    #graph.plot(figsize=(20,20))

    # initialize
    grammers = []
    lang = [0 for _ in range(city_size)]
    for i in range(city_size):
        grammers.append(generate_init_matrix(i,city_size))
    #base = data.plot(column='language' ,cmap = 'rainbow',figsize=(20,20))
    #fig = plt.show()

    print("| 文法行列(grammers):{}*{}".format(grammers[0].shape,len(grammers)))
    print("| 言語(lang):{}".format(lang))


    for epoch in range(100):
        print("\n| epoch:{}".format(epoch))
        #for i in range(city_size):
        #    grammers[i] = update_matrix(grammers,line_list,i)
        grammers = update_matrix(grammers,line_list)


        lang = gpd_language(data,grammers,lang)
        #print(data.language)
        print("| 言語(lang):{}".format(len(lang)))
        print("| 文法行列(grammers):{}*{}".format(grammers[0].shape,len(grammers)))

        base = data.plot(column='language' ,cmap = 'rainbow',figsize=(20,20))
        #fig = plt.show()
        #fig = plt.figure()
        plt.savefig("img_{}.png".format(epoch))

    base = data.plot(column='language' ,cmap = 'rainbow',figsize=(20,20))
    fig = plt.show()

main()