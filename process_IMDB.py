import pandas as pd
import numpy as np
import scipy as sp
import torch
from sklearn.feature_extraction.text import CountVectorizer
from preprocess_DBLP import *

movies = pd.read_csv('data/movie_metadata.csv', encoding='utf-8').dropna(
    axis=0, subset=['actor_1_name', 'director_name']).reset_index(drop=True)

# extract labels
label_to_idx_dict = {}
labels_dict = []
i = 0
for movie_idx, genres in movies['genres'].items():
    label_dict = {}
    for genre in genres.split('|'):
        if genre not in label_to_idx_dict.keys():
            label_to_idx_dict[genre] = i
            i += 1
        label_dict[genre] = 1
    labels_dict.append(label_dict)

# convert label dicts to vectors
movie_labels = []
num_labels = len(label_to_idx_dict.keys())
for i in range(len(labels_dict)):
    label = [0] * num_labels
    for genre in labels_dict[i].keys():
        label[label_to_idx_dict[genre]] = 1
    movie_labels.append(label)

# remove labels that don't have any nodes in which they are most prominent (15, 18, 21, 22, 23)
movie_labels = np.array(movie_labels)
movie_labels = np.hstack((movie_labels[:, :15], movie_labels[:, 16:18], movie_labels[:, 19:21]))
num_labels = movie_labels.shape[1]

# movie feature assignment
# extract bag-of-word representations of plot keywords for each movie
# X is a sparse matrix
vectorizer = CountVectorizer(min_df=2)
movie_X = vectorizer.fit_transform(movies['plot_keywords'].fillna('').values)

# get director list and actor list
directors = list(set(movies['director_name'].dropna()))
directors.sort()
actors = list(set(movies['actor_1_name'].dropna().to_list() +
                  movies['actor_2_name'].dropna().to_list() +
                  movies['actor_3_name'].dropna().to_list()))
actors.sort()

# generate edges
movie_director = [[], []]
movie_actor = [[], []]
for movie_idx, row in movies.iterrows():
    if row['director_name'] in directors:
        director_idx = directors.index(row['director_name'])
        movie_director[0].append(movie_idx)
        movie_director[1].append(director_idx)
    if row['actor_1_name'] in actors:
        actor_idx = actors.index(row['actor_1_name'])
        movie_actor[0].append(movie_idx)
        movie_actor[1].append(actor_idx)
    if row['actor_2_name'] in actors:
        actor_idx = actors.index(row['actor_2_name'])
        movie_actor[0].append(movie_idx)
        movie_actor[1].append(actor_idx)
    if row['actor_3_name'] in actors:
        actor_idx = actors.index(row['actor_3_name'])
        movie_actor[0].append(movie_idx)
        movie_actor[1].append(actor_idx)

movie_director = torch.tensor(movie_director)
movie_director_dict = edge_index_to_dict(movie_director)
director_movie_dict = edge_index_to_dict(torch.vstack((movie_director[1, :], movie_director[0, :])))
movie_actor = torch.tensor(movie_actor)
movie_actor_dict = edge_index_to_dict(movie_actor)
actor_movie_dict = edge_index_to_dict(torch.vstack((movie_actor[1, :], movie_actor[0, :])))

# print(director_movie_dict)
# print(len(director_movie_dict))
# print(len(directors))
# print(movie_director_dict)
# print(max(movie_director_dict.values(), key=len))

edge_index = [[], []]
dir_labels = np.zeros((len(directors), num_labels))
dir_features = np.zeros((len(directors), movie_X.shape[1]))
for director, movies in director_movie_dict.items():
    for movie in movies:
        dir_labels[director] += movie_labels[movie]
        dir_features[director] += movie_X[movie]
        for actor in movie_actor_dict[movie]:
            for movie2 in actor_movie_dict[actor]:
                for end_dir in movie_director_dict[movie2]:
                    edge_index[0].append(director)
                    edge_index[1].append(end_dir)

for i in range(dir_labels.shape[0]):
    dir_labels[i] = dir_labels[i] / sum(dir_labels[i])

dir_features = torch.tensor(dir_features)
dir_features[dir_features > 1] = 1

# print("director features")
# print(dir_features)
# print(dir_features.shape)
# print("director labels")
# print(dir_labels)
# print(dir_labels.shape)
# print("director edges")
# print(movie_director)
# print(len(movie_director[0]))

np.savez("data/imdb/imdb_graph", x=dir_features, y=dir_labels, edge_index=edge_index)

imdb_file = np.load("data/imdb/imdb_graph.npz")
imdb = geo.data.Data()
imdb.x = torch.tensor(imdb_file["x"])
imdb.y = torch.tensor(imdb_file["y"])
imdb.edge_index = torch.tensor(imdb_file["edge_index"], dtype=torch.int64)

display_graph_stats(extract_cc(imdb, save=True, file_name="data/imdb/imdb_cc"))

# Metapath: Director Movie Actor Movie Director, labels and features extracted by the first movie connection to Director
