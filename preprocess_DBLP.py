import torch_geometric as geo
import torch
from collections import defaultdict
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
import copy

def parse_node(path, sep='\t'):
    """
    change a node file to two mappings. One containing all the id's in order, and another
    containing the reverse of the former (so index - id and id - index)
    :param path: a string containing the path to the node file to be parsed
    :param sep: the string used to separate columns
    :return: an array with the index-id pairs and dictionary with the id-index pairs
    """
    file = open(path)
    index = []
    id = {}
    i = 0
    for line in file.readlines():
        line = line.strip('\n').split(sep)
        index.append(int(line[0]))
        id[int(line[0])] = i
        i += 1
    return index, id

def parse_two_col(path, unique=True, sep='\t'):
    """
    returns two dictionaries with mappings of the first column to the second and vice versa (for .txt files)
    :param path: a string containing the path of the file to be parsed
    :param unique: if each key will only have one value
    :param sep: the string used to separate columns
    :return: two dictionaries mapping the first column to the second and vice versa
    """
    file = open(path)
    if unique:
        forward = {}
        backward = {}
    else:
        forward = defaultdict(list)
        backward = defaultdict(list)
    for line in file.readlines():
        line = line.strip('\n').split(sep)
        first = int(line[0])
        second = int(line[1])
        if unique:
            forward[first] = second
            backward[second] = first
        else:
            forward[first].append(second)
            backward[second].append(first)

    return forward, backward

def parse_two_col_dat(path: str):
    """
    returns two dictionaries with mappings of the first column to the second and vice versa (for .dat files)
    :param path: a string containing the path of the file to be parsed
    :return: two dictionaries mapping the first column to the second and vice versa
    """
    file = open(path)
    forward = defaultdict(list)
    backward = defaultdict(list)
    for line in file:
        line = line.split()
        first = int(line[0])
        second = int(line[1])
        forward[first].append(second)
        backward[second].append(first)
    return forward, backward

def edge_index_to_dict(edge_index):
    """
    helper function that converts an edge_index tensor into a dictionary
    :param edge_index: a 2 x num_edges tensor that represents the edges in a graph
    :return: a dictionary that represents has each node as a key, and each value is a list of
                the nodes it is connected to
    """
    #num_edges = edge_index
    #print(num_edges)
    edge_dict = defaultdict(list)
    #print(edge_index)
    num_edges = len(edge_index[0])
    for i in range(num_edges):
        edge_dict[edge_index[0][i].item()].append(edge_index[1][i].item())
    return edge_dict

def edge_dict_to_index(edge_dict: dict):
    """
    helper function that converts edges stored as a dictionary into a num_edges x 2 tensor representing the edges
    :param edge_dict: the edge dictionary to be converted
    :return: a num_edges x 2 tensor representing the edges of a graph
    """
    edge_index = [[], []]
    for start, end_list in edge_dict.items():
        for end in end_list:
            edge_index[0].append(start)
            edge_index[1].append(end)
    return torch.tensor(edge_index)


def hetero_to_homo_distribution(graph: geo.data.HeteroData, metapath: tuple, dist_source: dict, num_dist: int,num_nodes:int,author_dict:dict):
    '''
    changes a heterogeneous graph into a homogenous graph based on a metapath
    ** note that the format of the inputted graph is as follows, edges should be stored under
        graph_name[node_type1, node_type2].edge_index
    :param graph: the input heterogeneous graph
    :param metapath: a tuple of strings containing the path to follow in order to homogenize the graph
                    * should be symmetric *
    :param dist_source: a dictionary mapping the nodes in the middle of the metapath to the nodes we want the label distributions to be pulled from
    :param num_dist: the number of different terms to be stored in the distribution
    :return: a homogenous graph
    '''
    num_edges = len(metapath) - 1
    edge_dict_list = []
    label_dist = defaultdict(lambda: [0] * num_dist)
    # creates each step's edge dictionaries in the metapath
    for i in range(num_edges):
        edge_index = graph[metapath[i], 'to', metapath[i + 1]].edge_index
        #edge_dict = edge_index_to_dict(edge_index)
        edge_dict = edge_index
        edge_dict_list.append(edge_dict)
    # condenses the edge dictionaries into one final edge dictionary
    while len(edge_dict_list) > 1:
        # create the label distribution
        if len(edge_dict_list) == num_edges//2 + 1:
            for start, end_list in edge_dict_list[0].items():
                #print("end_list",end_list)
                for end in end_list:
                    #print(end)
                    #print(dist_source)
                    #print(dist_source[end])
                    label_dist[start][dist_source[end]] += 1
        combined_edge_dict = defaultdict(list)
        for start, end_list in edge_dict_list[0].items():
            for end in end_list:
                combined_edge_dict[start].extend(edge_dict_list[1][end])
        # remove the first two dictionaries and replaced with the combined
        edge_dict_list.pop(0)
        edge_dict_list.pop(0)
        edge_dict_list.insert(0, combined_edge_dict)
    new_edges = edge_dict_to_index(edge_dict_list[0])
    # fix edge indexes by converting to an adjacency matrix and removing self connections
    #adj = geo.utils.to_dense_adj(new_edges)
    #adj[adj != 0] = 1
    #adj = adj.squeeze()
    #num_nodes = adj.size(dim=0)
    #adj = torch.sub(adj, torch.eye(num_nodes))
    #new_edges = geo.utils.dense_to_sparse(adj)[0]
    # normalize the distribution
    dist_tens = torch.empty(num_nodes, num_dist)
    for node, dist in label_dist.items():
        tot = sum(dist)
        label_dist[node] = [x / tot for x in dist]
        dist_tens[author_dict[node], :] = torch.tensor(label_dist[node])
    new_graph = geo.data.Data()
    new_graph.x = graph[metapath[-1]].x
    new_graph.y = dist_tens
    new_graph.edge_index = new_edges
    return new_graph


def hetero_to_homo(graph: geo.data.HeteroData, metapath: tuple):
    """
    changes a heterogeneous graph into a homogenous graph based on a metapath
    ** note that the format of the inputted graph is as follows, edges should be stored under
        graph_name[node_type1, node_type2].edge_index
    :param graph: the input heterogeneous graph
    :param metapath: a tuple of strings containing the path to follow in order to homogenize the graph
                    * should be symmetric *
    :return: a homogenous graph
    """
    num_edges = len(metapath) - 1
    edge_dict_list = []
    # creates each step's edge dictionaries in the metapath
    for i in range(num_edges):
        edge_index = graph[metapath[i], 'to', metapath[i + 1]].edge_index
        #print(edge_index)
        #edge_dict = edge_index_to_dict(edge_index)
        edge_dict = edge_index
        edge_dict_list.append(edge_dict)
    # condenses the edge dictionaries into one final edge dictionary
    while len(edge_dict_list) > 1:
        combined_edge_dict = defaultdict(list)
        for start, end_list in edge_dict_list[0].items():
            for end in end_list:
                combined_edge_dict[start].extend(edge_dict_list[1][end])
        # remove the first two dictionaries and replaced with the combined
        edge_dict_list.pop(0)
        edge_dict_list.pop(0)
        edge_dict_list.insert(0, combined_edge_dict)
    new_edges = edge_dict_to_index(edge_dict_list[0])
    print(new_edges)
    # fix edge indexes by converting to an adjacency matrix and removing self connections
    #adj = geo.utils.to_dense_adj(new_edges)
    #adj[adj != 0] = 1
    #adj = adj.squeeze()
    #num_nodes = adj.size(dim=0)
    #adj = torch.sub(adj, torch.eye(num_nodes))
    #new_edges = geo.utils.dense_to_sparse(adj)[0]
    new_graph = geo.data.Data()
    #print("feature is:",meta_path[-1])
    new_graph.x = graph[metapath[-1]].x
    new_graph.edge_index = new_edges
    return new_graph

def display_graph_stats(data, number_label=False):
    """
    Displays graph statistics
    :param data: the graph to display
    :param number_label: whether you want to have the actual number labels of the bar graph
    :return: None
    """
    print(data)
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    net_data = geo.utils.convert.to_networkx(data).to_undirected()  # convert torch graph to networkx
    is_con = nx.is_connected(net_data)  # if the graph is connected
    print(f'Is connected: {is_con}')
    con_comps = [len(c) for c in sorted(nx.connected_components(net_data), key=len, reverse=True)]
    print(f'Number of connected components: {len(con_comps)}')
    print(f'Connected components: {con_comps}')

    # degree distribution
    adj = geo.utils.to_dense_adj(data.edge_index)
    adj = adj.squeeze()
    freq_dict = defaultdict(int)
    freq, _ = torch.sort(torch.sum(adj, dim=0))
    freq = freq.squeeze()
    for f in freq:
        freq_dict[f.item()] += 1
    freq_x = []
    freq_y = []
    for deg, f in freq_dict.items():
        freq_x.append(deg)
        freq_y.append(f)
    freq_x = np.array(freq_x)
    freq_y = np.array(freq_y)



    # graphs
    fig1, a = plt.subplots()
    a.plot(freq_x, freq_y)
    a.set(xlabel='Degree', ylabel='Frequency', title="Degree Distribution")

    # Label Distribution
    size_dist = data.y.size(dim=1)

    # find the distributions that have the i-th label as the highest value in the distribution
    dist = data.y
    filtered_dist = defaultdict(list)
    for i in range(dist.size(dim=0)):
        curr_dist = dist[i, :]
        filtered_dist[torch.argmax(curr_dist).item()].append(list(curr_dist))
    for k, v in filtered_dist.items():
        filtered_dist[k] = torch.tensor(v)

    fig2, ax = plt.subplots(len(filtered_dist.keys()) + 1, sharex=True)
    fig2.set_figheight((len(filtered_dist.keys()) + 1) * 2)
    ax[0].plot(np.array(list(range(size_dist))), np.array(torch.transpose(data.y, 0, 1)))
    ax[0].set(ylabel='Probability', title="Label Distributions")
    i = 1
    for k in filtered_dist.keys():
        avg_dist = torch.mean(filtered_dist[k], dim=0)
        avg_dist = np.array(avg_dist)
        ax[i].bar(np.array(list(range(size_dist))), avg_dist)
        ax[i].set(ylabel='Probability', title=f'Average Label Distribution for nodes of mainly {k}, using {filtered_dist[k].size(dim=0)} nodes')
        # Add annotation to bars
        if number_label:
            for x in range(avg_dist.size):
                ax[i].text(x - .3, 0.5,
                         str(avg_dist[x]),
                         fontsize=10, fontweight='bold',
                         color='black')
        i += 1
    ax[-1].set(xlabel='Distribution Labels')
    fig2.suptitle("Graph's Label Distributions")

    # node by node difference of first and second most dominant distribution label
    fig3, ax1 = plt.subplots(2)
    sorted_dist, sorted_dist_idx = torch.sort(dist, descending=True)
    diff = []
    combos = set()
    for i in range(dist.size(dim=0)):
        diff_num = (sorted_dist[i][0] - sorted_dist[i][1]).item()
        if diff_num < .1:
            combos.add((sorted_dist_idx[i][0].item(), sorted_dist_idx[i][1].item()))
        diff.append(diff_num)
    print(f'There are {len(combos)} combinations of top the two most prominent labels that have a difference of .1 or less: {combos}')
    ax1[0].bar(list(range(dist.size(dim=0))), diff)
    sorted_diff = sorted(diff)
    ax1[1].bar(list(range(dist.size(dim=0))), sorted_diff)
    for i in range(len(sorted_diff)):
        if sorted_diff[i] > .1:
            print(f'There are this many nodes with a difference below .1 {i}')
            break
    fig3.suptitle("Difference in probability between the first and second most prominent labels")
    ax1[0].set(ylabel='Probability')
    ax1[1].set(xlabel='Nodes', ylabel='Probability', title='Sorted by difference')

    plt.show()
    # print actual data
    for key, value in data.to_dict().items():
        print(f'{key}: {value}')

def create_masks(data: geo.data.Data, train_perc: float, val_perc: float, test_perc: float, random_state: int = None, file_name: str = "data/masks", save=False):
    """
    returns a training, validation, and testing mask based on the inputted percentages, and saves them into a npz file
    under the names: train_mask, val_mask, test_mask. Also saves the real percentages under the name true_perc
    Note that train_perc + val_perc + test_perc should add up to 1.00
    :param data: the graph you want to generate masks for
    :param train_perc: the percentage of the graph you want your training mask to be
    :param val_perc: the percentage of the graph you want your validation mask to be
    :param test_perc: the percentage of the graph you want your test mask to be
    :param random_state: an int to seed the random mask generation
    :param file_name: the name of the file you want to save the masks to
    :param save: whether or not to save the generated masks to a npz file
    :return: the masks and a list containing their true percentages
    """
    num_nodes = data.x.size(dim=0)
    idx_list = list(range(num_nodes))
    train_mask, val_test = train_test_split(idx_list, train_size=train_perc, random_state=random_state)
    val_mask, test_mask = train_test_split(val_test, train_size=val_perc/(1 - train_perc), random_state=random_state)

    # convert indices to boolean masks
    bool_tens = torch.zeros(num_nodes)
    bool_tens[train_mask] = 1
    train_mask = bool_tens.to(torch.bool)

    bool_tens = torch.zeros(num_nodes)
    bool_tens[val_mask] = 1
    val_mask = bool_tens.to(torch.bool)

    bool_tens = torch.zeros(num_nodes)
    bool_tens[test_mask] = 1
    test_mask = bool_tens.to(torch.bool)

    true_perc = [train_mask.sum()/num_nodes, val_mask.sum()/num_nodes, test_mask.sum()/num_nodes]

    if save:
        np.savez(file_name, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, true_perc=true_perc)

    return train_mask, val_mask, test_mask, true_perc

def extract_cc(data, save=False, file_name=""):
    """
    returns the largest connected component of data graph
    :param data: the graph you want to extract the largest connected component from
    :param save: whether you want to save the largest cc or not
    :param file_name: the file name you want to save the largest cc to
    :return: the largest connected component of data
    """
    net_data = geo.utils.convert.to_networkx(data).to_undirected()
    largest_cc = max(nx.connected_components(net_data), key=len)
    largest_cc = data.subgraph(torch.tensor(list(largest_cc)))
    if save:
        np.savez(file_name, x=largest_cc.x, y=largest_cc.y, edge_index=largest_cc.edge_index)
    return largest_cc

def display_label_stats(data: geo.data.Data, data_name: str):
    """
    generate the stats on each label distribution (std dev, average distribution)
    :param data: the graph that you want to generate statistics from
    :param data_name: the name of the dataset
    :return: the stats mentioned above
    """
    # find the distributions that have the i-th label as the highest value in the distribution
    dist = data.y
    filtered_dist = defaultdict(list)
    for i in range(dist.size(dim=0)):
        curr_dist = dist[i, :]
        filtered_dist[torch.argmax(curr_dist).item()].append(list(curr_dist))
    for k, v in filtered_dist.items():
        filtered_dist[k] = torch.tensor(v)
    std_list = [0] * len(filtered_dist.keys())
    mean_list = [0] * len(filtered_dist.keys())
    for k, v in filtered_dist.items():
        # the average standard deviation of the distributions with k as the most prominent label
        std_list[k] = torch.std(v, dim=0)
        mean_list[k] = torch.mean(filtered_dist[k], dim=0)
    dis_list_string = ""
    for i in range(len(std_list)):
        for j in range(len(std_list[i])):
            dis_list_string += f'{mean_list[i][j].item():.4f}' + " +- " \
                               + f"{std_list[i][j].item():.4f}" + ", "
        dis_list_string += "\n"
    print(dis_list_string)

    size_dist = len(mean_list[0])
    fig2, ax = plt.subplots(size_dist, sharex=True)  # len(filtered_dist.keys())
    fig2.set_figheight((len(filtered_dist.keys())) * 2)
    # ax[0].plot(np.array(list(range(1, len(mean_list) + 1))), np.array(torch.transpose(data.y, 0, 1)))
    # ax[0].set(ylabel='Probability', title=f"Label Distributions")
    i = 0
    for k in range(size_dist): # range(size_dist):
        avg_dist = mean_list[k]
        avg_dist = np.array(avg_dist)
        x_lab = []
        for num in range(1, 1 + size_dist):
            x_lab.append(f"Class {num}")
        ax[i].bar(np.array(x_lab), avg_dist, yerr=std_list[k])
        ax[i].set(ylabel='Probability')
                  # , title=f'Average Label Distribution for nodes of mainly {k + 1}, using {filtered_dist[k].size(dim=0)} nodes')
        for x in range(size_dist):
            ax[i].text(x, avg_dist[x],
                       f'{avg_dist[x]:.4f}', ha='center',
                       fontsize=10, fontweight='bold',
                       color='black') # + "+-" + f'{std_list[k][x].item():.4f}'
        i += 1

    ax[-1].set(xlabel='Distribution Labels')
    # fig2.suptitle(f"{data_name}'s Label Distributions")
    plt.savefig(data_name + "_label_dist_fig")
    plt.show()
    return std_list, mean_list

def id_to_index(edge_index,row_map,col_map):
    edge_index_new = copy.deepcopy(edge_index)
    for i in range(edge_index.shape[1]):
        edge_index_new[0][i] = row_map[edge_index[0][i].item()][0]
        edge_index_new[1][i] = col_map[edge_index[1][i].item()][0]
    return edge_index_new
# opening files
# nodes and their labels

author_arr, author_dict = parse_node("DBLP/author.txt")
author_label = parse_two_col("DBLP/author_label.txt")[0]
print("start")
#print(author_dict)
author_index = []
aid = author_label.keys()
# index rebuild from original index
for aid_ in aid:
    author_index.append(author_dict[aid_])
author_arr_new = []
for i in range(len(author_arr)):
    if i in author_index:
        author_arr_new.append(author_arr[i])
#print("new author list",len(author_arr_new))
author_dict_new = defaultdict(list)
for i in range(len(author_arr_new)):
    author_dict_new[author_arr_new[i]].append(i)

#print("author dict new",author_dict_new)

#paper_arr, paper_dict = parse_node("DBLP/paper.txt")
# only has 100 labels, not useful
#paper_label = parse_two_col("DBLP/paper_label.txt")[0]
term_arr, term_dict = parse_node("DBLP/term.txt")

conf_arr, conf_dict = parse_node("DBLP/conf.txt")
conf_label = parse_two_col("DBLP/conf_label.txt")[0]
# edges
paper_author, author_paper = parse_two_col("DBLP/paper_author.txt", unique=False)
paper_term,term_paper = parse_two_col("DBLP/paper_term.txt", unique=False)
paper_conf,conf_paper = parse_two_col("DBLP/paper_conf.txt", unique=False)
paper_author_new = copy.deepcopy(paper_author)
author_paper_new = copy.deepcopy(author_paper)
#print(author_arr_new)
for key,val in paper_author.items():
    for v in val:
        if not(v in author_arr_new):
            paper_author_new[key].remove(v)
        if len(paper_author_new[key])==0:
            paper_author_new.pop(key)
paper_id = paper_author_new.keys()
paper_term_new = copy.deepcopy(paper_term)
for key,val in paper_term.items():
    #print(key)
    if not(key in paper_id):
        paper_term_new.pop(key)
edge_index_paper_term = edge_dict_to_index(paper_term_new)
edge_index_term_paper = [edge_index_paper_term[1],edge_index_paper_term[0]]
term_paper_new = edge_index_to_dict(edge_index_term_paper)

paper_conf_new = copy.deepcopy(paper_conf)
for key,val in paper_conf.items():
    if not(key in paper_id):
        paper_conf_new.pop(key)
edge_index_paper_conf = edge_dict_to_index(paper_conf_new)
edge_index_conf_paper = [edge_index_paper_conf[1],edge_index_paper_conf[0]]
conf_paper_new = edge_index_to_dict(edge_index_conf_paper)
conf_id = conf_paper_new.keys()
term_id = term_paper_new.keys()
paper_dict_new = defaultdict(list)
paper_id = list(paper_id)
term_id = list(term_id)
conf_id = list(conf_id)
conf_dict_new = defaultdict(list)
for i in range(len(paper_id)):
    paper_dict_new[paper_id[i]].append(i)
term_dict_new = defaultdict(list)
for i in range(len(term_id)):
    term_dict_new[term_id[i]].append(i)
for i in range(len(conf_id)):
    conf_dict_new[conf_id[i]].append(i)
#print(len(paper_dict_new.keys()))



for key,val in author_paper.items():
    if not(key in author_arr_new):
        author_paper_new.pop(key)


num_authors = len(author_arr_new)
num_terms = len(term_arr)
#num_terms = len(term_id)
print("number of terms",num_terms)
# find what authors need to be connected to each other
author_edges = [[], []]
# will become bag of words description of authors' papers' keywords
feature_tens = torch.zeros(num_authors, num_terms)
#author_dist = defaultdict(lambda: [0] * 4)
for author, papers in author_paper_new.items():
    for paper in papers:
        for term in paper_term_new[paper]:
            # frequency of terms
            feature_tens[author_dict_new[author][0]][term_dict[term]] += 1
        # # add edges to the list
        # for con in paper_author[paper]:
        #     author_edges[0].append(author_dict[author])
        #     author_edges[1].append(author_dict[con])

# normalize the distribution and place it into a label list
#dist_tens = torch.empty(num_authors, 4)
#for author, dist in author_dist.items():
#    tot = sum(dist)
#    dist_tens[author, :] = torch.tensor([x / tot for x in dist])
#data = geo.datasets.DBLP(root='./DBLP/')
#hetero_graph = data[0]
edge_index_author_paper = edge_dict_to_index(author_paper_new)
print(".................")
#print(author_paper_new)
#print(edge_index_author_paper)
edge_index_paper_author = [edge_index_author_paper[1],edge_index_author_paper[0]]
paper_author_new = edge_index_to_dict(edge_index_paper_author)
#print(len(edge_index_paper_author[0]))
#print(feature_tens.shape)
hetero_graph = geo.data.HeteroData()
hetero_graph['author'].x = feature_tens
#hetero_graph['author'].y = dist_tens
hetero_graph['author','to','paper'].edge_index = id_to_index(edge_dict_to_index(author_paper_new),author_dict_new,paper_dict_new)
hetero_graph['paper','to','author'].edge_index = id_to_index(edge_dict_to_index(paper_author_new),paper_dict_new,author_dict_new)
hetero_graph['term','to','paper'].edge_index = id_to_index(edge_dict_to_index(term_paper_new),term_dict_new,paper_dict_new)
hetero_graph['paper','to','term'].edge_index = id_to_index(edge_dict_to_index(paper_term_new),paper_dict_new,term_dict_new)
hetero_graph['conference','to','paper'].edge_index = id_to_index(edge_dict_to_index(conf_paper_new),conf_dict_new,paper_dict_new)
hetero_graph['paper','to','conference'].edge_index = id_to_index(edge_dict_to_index(paper_conf_new),paper_dict_new,conf_dict_new)

print("edge index example:",hetero_graph['term','to','paper'])
#print(edge_dict_to_index(paper_author_new).shape)
#print(edge_dict_to_index(paper_term_new).shape)
#print(edge_dict_to_index(term_paper_new).shape)
#print(edge_dict_to_index(conf_paper_new).shape)
#print(edge_dict_to_index(paper_conf_new).shape)
meta_path_list = [('author','paper','author'),('author','paper','conference','paper','author'),('author','paper','term','paper','author')]
meta_path_list = [('author','paper','author')]
#print(author_paper)
print("------------------")
print("--------------------")
#print("paper conf:",edge_dict_to_index(paper_conf).shape)
#pc = edge_dict_to_index(paper_conf)
#print(len(set(pc[0])))
paper_label = copy.deepcopy(paper_conf)
num_nodes = feature_tens.shape[0]
#num_nodes = hetero_graph['author'].x.shape[0]
#print("num_nodes",num_nodes)
for p,c in paper_label.items():
    paper_label[p] = conf_label[c[0]]

homo_graph_list = []
count = 0
name_list = ['APA','APCPA','APTPA']
'''for meta_path in meta_path_list:
    if count == 0:
        homo_graph_list.append(hetero_to_homo_distribution(hetero_graph,meta_path,paper_label,4,num_nodes,author_dict_new))
    else:
        homo_graph_list.append(hetero_to_homo(hetero_graph,meta_path))

    #print(homo_graph_list[count])
    #np.savez("dblp_homo_list/"+name_list[count], x=homo_graph_list[count].x, y=homo_graph_list[0].y, edge_index=homo_graph_list[count].edge_index)
    count+=1
'''
y = np.load('./dblp_homo_list/APA.npz')['y']
hetero_graph['author'].y = torch.from_numpy(y)
print(y.shape)
print(hetero_graph['author'].num_nodes)
#for edge_type,store in hetero_graph.edge_items():
#    print(edge_type)
#    print(store.edge_index)

#torch.save(hetero_graph,"./DBLP/hg.pt")
# fix edge indexes by converting to an adjacency matrix and removing self connections
# author_edges = torch.tensor(author_edges)
# adj = geo.utils.to_dense_adj(author_edges)
# adj[adj != 0] = 1
# adj = adj.squeeze()
# adj = torch.sub(adj, torch.eye(num_authors))
# author_edges = geo.utils.dense_to_sparse(adj)[0]

# homo_graph = geo.data.Data(x=feature_tens, edge_index=author_edges, y=dist_tens)

# print(homo_graph)
# print(f'Author-Author Features: {homo_graph.x}')
# print(f'Author-Author Edge Indices: {homo_graph.edge_index}')
# # note that DBLP doesn't have the paper labels so topic label distributions just end up being 1-1-1-1
# print(f'Author topic label distribution: {homo_graph.y}')

def return_graph():
    return hetero_graph

