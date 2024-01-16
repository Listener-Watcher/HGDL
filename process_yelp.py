#from preprocess_DBLP import *
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import networkx as nx
import torch
from collections import defaultdict
import torch_geometric as geo
# converting json files into lists of dicts
# business = []
# with open('data/yelp_dataset/yelp_academic_dataset_business.json', encoding='utf-8') as business_file:
#     for line in business_file:
#         business.append(json.loads(line))
# print("done with business")
# checkin = []
# with open('data/yelp_dataset/yelp_academic_dataset_checkin.json', encoding='utf-8') as checkin_file:
#     for line in checkin_file:
#         checkin.append(json.loads(line))
# print("done with checkin")
# review = []
# with open('data/yelp_dataset/yelp_academic_dataset_review.json', encoding='utf-8') as review_file:
#     for line in review_file:
#         review.append(json.loads(line))
# np.save("data/yelp/review", review)
# print("done with review")
# tip = []
# with open('data/yelp_dataset/yelp_academic_dataset_tip.json', encoding='utf-8') as tip_file:
#     for line in tip_file:
#         tip.append(json.loads(line))
# np.save("data/yelp/tip", tip)
# print("done with tip")
# user = []
# with open('data/yelp_dataset/yelp_academic_dataset_user.json', encoding='utf-8') as user_file:
#     for line in user_file:
#         user.append(json.loads(line))
# np.save("data/yelp/user", user)
# print("done with user")
# np.save("data/yelp/business", business)
# np.save("data/yelp/checkin", checkin)

# 150,346 businesses
# 1311 categories
'''
business_list = np.load("data/yelp/business.npy", allow_pickle=True)
business_cat = []
index_to_business = []
business_to_index = {}
categories = set()
for business in business_list:
    business_to_index[business["business_id"]] = len(index_to_business)
    index_to_business.append(business["business_id"])
    bus_cat = []
    if business["categories"]:
        bus_cat += business["categories"].split(", ")
        categories.update(bus_cat)
    business_cat.append(bus_cat)

categories = list(categories)
num_bus = business_list.shape[0]
print(num_bus)
business_label = torch.zeros(num_bus, len(categories))
for i in range(num_bus):
    for cat in business_cat[i]:
        business_label[i][categories.index(cat)] += 1

np.save("data/yelp/business_label", business_label)

np.savez("data/yelp/business_id", index_to_business=index_to_business, business_to_index=business_to_index)
'''

"""business_list = np.load("data/yelp/business.npy", allow_pickle=True)

business_id = np.load("data/yelp/business_id.npz", allow_pickle=True)
index_to_business = business_id["index_to_business"]
business_to_index = business_id["business_to_index"]

business_label = np.load("data/yelp/business_label.npy", allow_pickle=True)
business_to_index = business_to_index.item()

num_bus = len(index_to_business)
print(num_bus)
review_list = np.load("data/yelp/review.npy", allow_pickle=True)
business_text = [""] * num_bus
for review in review_list:
    bus_idx = business_to_index[review["business_id"]]
    business_text[bus_idx] = ' '.join((business_text[bus_idx], review["text"]))


CountVecBusReviews = CountVectorizer(ngram_range=(1, 1), stop_words='english')
count_review_data = CountVecBusReviews.fit_transform(business_text)
print(count_review_data[:4, :])
np.save("data/yelp/business_feat", count_review_data.toarray())

print("saved features and labels")"""



def make_yelp_subset_bus(file_name, num_elem, save=True):
    """
    creates a yelp subset (user - business - user) of num_elem users and saves it under file_name with keywords
    "x", "y", and "edge_index"
    :param file_name: the name of the file to save the subset graph of yelp to
    :param num_elem: the number of nodes you approximately want in your subset
    :param save: whether you want to save the generated subset
    :return: x, y, edge_index
    """
    user_list = np.load("data/yelp/user.npy", allow_pickle=True)
    user_id = np.load("data/yelp/user_id.npz", allow_pickle=True)
    user_feat = np.load("data/yelp/user_feat.npy", allow_pickle=True)

    index_to_user = user_id["index_to_user"]
    user_to_index = user_id["user_to_index"].item()

    business_list = np.load("data/yelp/business.npy", allow_pickle=True)
    business_id = np.load("data/yelp/business_id.npz", allow_pickle=True)
    index_to_business = business_id["index_to_business"]
    business_to_index = business_id["business_to_index"].item()
    review_list = np.load("data/yelp/review.npy", allow_pickle=True)

    bus_subset_idx = np.random.choice(len(index_to_business), num_elem, replace=False)

    bus_subset_id = {}
    full_to_sub = {}
    i = 0
    for bus in bus_subset_idx:
        # subset_bus_feat.append(user_feat[user])
        bus_subset_id[index_to_business[bus]] = bus
        full_to_sub[bus] = i
        i += 1

    star_dist = [defaultdict(int) for _ in range(len(bus_subset_idx))]
    star_set = set()
    user_to_bus_dict = defaultdict(set)
    subset_bus_text = [""] * num_elem
    for review in review_list:
        if review["user_id"] in user_to_index.keys() and review["business_id"] in bus_subset_id.keys():
            bus_idx = full_to_sub[bus_subset_id[review["business_id"]]]
            user_to_bus_dict[review["user_id"]].add(bus_idx)
            star_dist[bus_idx][review["stars"]] += 1
            star_set.add(review["stars"])
            # aggregating the business review text
            subset_bus_text[bus_idx] += review["text"]

    # changing string aggregation of the businesses to bag of words
    CountVecBusReviews = CountVectorizer(ngram_range=(1, 1), stop_words='english', min_df=50)
    subset_bus_feat = CountVecBusReviews.fit_transform(subset_bus_text)
    subset_bus_feat = subset_bus_feat.toarray()
    print(f'bus feat shape: {subset_bus_feat.shape}')
    print(subset_bus_feat[:4, :])

    print(f'number of users checked {len(user_to_bus_dict.keys())}')

    edge_index = [[], []]
    for bus_set in user_to_bus_dict.values():
        bus_set = list(bus_set)
        for i in range(len(bus_set) - 1):
            for j in range(i + 1, len(bus_set)):
                edge_index[0].append(bus_set[i])
                edge_index[1].append(bus_set[j])
    # fix edge_index by removing repeats and self edges
    edge_index = torch.tensor(edge_index)
    adj = geo.utils.to_dense_adj(edge_index)
    adj = adj.squeeze()
    num_nodes = adj.size(dim=0)
    adj[adj != 0] = 1
    edge_index = geo.utils.dense_to_sparse(adj)[0]

    star_dist_final = np.zeros((len(bus_subset_idx), len(star_set)))
    for i in range(len(bus_subset_idx)):
        for k, v in star_dist[i].items():
            star_dist_final[i][int(k) - 1] = v
        tot = np.sum(star_dist_final[i, :])
        star_dist_final[i, :] = star_dist_final[i, :] / tot

    if save:
        np.savez(file_name, x=subset_bus_feat.astype("float64"), y=star_dist_final.astype("float64"), edge_index=edge_index)

    return np.array(subset_bus_feat), star_dist_final, edge_index


def make_yelp_subset_cus(file_name, num_elem, save=True):
    """
    creates a yelp subset (user - business - user) of num_elem users and saves it under file_name with keywords
    "x", "y", and "edge_index"
    :param file_name: the name of the file to save the subset graph of yelp to
    :param num_elem: the number of nodes you approximately want in your subset
    :param save: whether you want to save the generated subset
    :return: x, y, edge_index
    """
    user_list = np.load("data/yelp/user.npy", allow_pickle=True)
    user_id = np.load("data/yelp/user_id.npz", allow_pickle=True)
    user_feat = np.load("data/yelp/user_feat.npy", allow_pickle=True)

    index_to_user = user_id["index_to_user"]
    user_to_index = user_id["user_to_index"].item()

    user_subset = set()
    while len(user_subset) <= num_elem:
        idx = np.random.choice(len(index_to_user))
        user_subset.add(idx)
        for friend_id in user_list[idx]["friends"].split(", "):
            if friend_id in user_to_index.keys():
                user_subset.add(user_to_index[friend_id])

    print(f'user subset has size: {len(user_subset)}')

    user_subset = list(user_subset)
    subset_user_id = {}
    subset_user_feat = []
    full_to_sub = {}
    i = 0
    for user in user_subset:
        subset_user_feat.append(user_feat[user])
        subset_user_id[index_to_user[user]] = user
        full_to_sub[user] = i
        i += 1

    business_list = np.load("data/yelp/business.npy", allow_pickle=True)
    business_id = np.load("data/yelp/business_id.npz", allow_pickle=True)
    index_to_business = business_id["index_to_business"]
    business_to_index = business_id["business_to_index"].item()
    review_list = np.load("data/yelp/review.npy", allow_pickle=True)

    cat_dist = [defaultdict(int) for _ in range(len(user_subset))]
    cat_dict = defaultdict(int)
    bus_to_user_dict = defaultdict(set)
    for review in review_list:
        if review["user_id"] in subset_user_id.keys() and review["business_id"] in business_to_index.keys():
            bus_idx = business_to_index[review["business_id"]]
            user_idx = full_to_sub[subset_user_id[review["user_id"]]]
            bus_to_user_dict[review["business_id"]].add(user_idx)
            bus = business_list[bus_idx]
            cat_dict[user_idx[str(bus["stars"])]]+=1
            car_dict[str(bus["stars"])]+=1
    print(f'number of businesses checked {len(bus_to_user_dict.keys())}')

    edge_index = [[], []]
    for user_set in bus_to_user_dict.values():
        user_set = list(user_set)
        for i in range(len(user_set) - 1):
            for j in range(i + 1, len(user_set)):
                edge_index[0].append(user_set[i])
                edge_index[1].append(user_set[j])
    # fix edge_index by removing repeats and self edges
    edge_index = torch.tensor(edge_index)
    adj = geo.utils.to_dense_adj(edge_index)
    adj = adj.squeeze()
    num_nodes = adj.size(dim=0)
    adj[adj != 0] = 1
    edge_index = geo.utils.dense_to_sparse(adj)[0]

    cat_list = [cat[0] for cat in sorted(cat_dict.items(), key=lambda x: x[1])[:40]]
    cat_to_index = {}
    i = 0
    for cat in cat_list:
        cat_to_index[cat] = i
        i += 1

    cat_dist_final = np.zeros((len(user_subset), len(cat_list)))
    print(f'there are {len(cat_list)} categories')
    for i in range(len(user_subset)):
        for k, v in cat_dist[i].items():
            if k in cat_list:
                cat_dist_final[i][cat_to_index[k]] = v
        tot = np.sum(cat_dist_final[i, :])
        cat_dist_final[i, :] = cat_dist_final[i, :] / tot
        if i < 3:
            print(cat_dist_final[i, :])

    if save:
        np.savez(file_name, x=subset_user_feat, y=cat_dist_final, edge_index=edge_index)

    return np.array(subset_user_feat), cat_dist_final, edge_index


'''
x, y, edge_index = make_yelp_subset_bus("data/yelp/yelp_bus_2", 3000, save=True)
yelp_sub = geo.data.Data(x=torch.tensor(x).type(dtype=torch.float), y=torch.tensor(y), edge_index=edge_index)
display_graph_stats(yelp_sub)

# yelp_subset_file = np.load("data/yelp/test_bus_subgraph.npz", allow_pickle=True)
# yelp_sub = geo.data.Data()
# yelp_sub.x = torch.tensor(yelp_subset_file["x"])
# yelp_sub.y = torch.tensor(yelp_subset_file["y"])
# yelp_sub.edge_index = torch.tensor(yelp_subset_file["edge_index"])
#
# display_graph_stats(yelp_sub)
#
# net_data = geo.utils.convert.to_networkx(yelp_sub).to_undirected()
# largest_cc = max(nx.connected_components(net_data), key=len)
# largest_cc = yelp_sub.subgraph(torch.tensor(list(largest_cc)))
# np.savez("data/yelp/test_bus_largest_cc", x=largest_cc.x, y=largest_cc.y, edge_index=largest_cc.edge_index)

'''
"""
# user stuff

user_list = np.load("data/yelp/user.npy", allow_pickle=True)
user_features = []
index_to_user = []
user_to_index = {}
for user in user_list:
    user_feat = []
    for k, v in user.items():
        if k == "user_id":
            user_to_index[v] = len(index_to_user)
            index_to_user.append(v)
        elif k == "name":
            continue
        elif k == "yelping_since":
            # date, probably need to split it
            user_feat.append(v)
        elif k == "friends" or k == "elite":
            user_feat.append(len(v))
        else:
            user_feat.append(v)
    user_features.append(user_feat)
np.savez("data/yelp/user_id", index_to_user=index_to_user, user_to_index=user_to_index)
np.save("data/yelp/user_feat", user_features)
"""
'''
# 1,987,897 users
user_list = np.load("data/yelp/user.npy", allow_pickle=True)
user_id = np.load("data/yelp/user_id.npz", allow_pickle=True)
user_feat = np.load("data/yelp/user_feat.npy", allow_pickle=True)
index_to_user = user_id["index_to_user"]
user_to_index = user_id["user_to_index"]
user_to_index = user_to_index.item()
# print(type(user_list[0]["friends"]))
#
print(f'number of users: {len(index_to_user)}')
#
user_subset = set()
while len(user_subset) <= 3000:
    idx = np.random.choice(len(index_to_user))
    user_subset.add(idx)
    for friend_id in user_list[idx]["friends"].split(", "):
        if friend_id in user_to_index.keys():
            user_subset.add(user_to_index[friend_id])
print(f'user subset has size: {len(user_subset)}')
# # user subset is 3025 users
#
user_subset = list(user_subset)
user_id_subset = {}
user_feat_subset = []
for user in user_subset:
    user_feat_subset.append(user_feat[user])
    user_id_subset[index_to_user[user]] = user
#
#
np.savez("data/yelp/user_subset", user_feat=user_feat_subset, user_id=user_id_subset, user=user_subset)
'''
'''
user_subset = np.load("data/yelp/user_subset.npz", allow_pickle=True)
user_feat = user_subset["user_feat"].tolist()
user_id = user_subset["user_id"].item()
users = user_subset["user"]
mask = np.ones(len(user_feat[0]), dtype=bool)
mask[1] = False
for i in range(len(user_feat)):
    user_feat[i] = np.array(user_feat[i])[mask]
    for j in range(user_feat[i].size):
        user_feat[i][j] = float(user_feat[i][j])
    if i < 10:
        print(user_feat[i])
np.savez("data/yelp/user_subset", user_feat=user_feat, user_id=user_id, user=users)
'''

user_subset = np.load("data/yelp/user_subset.npz", allow_pickle=True)
user_feat = user_subset["user_feat"].astype(float)
user_id = user_subset["user_id"].item()
users = user_subset["user"]

business_list = np.load("data/yelp/business.npy", allow_pickle=True)
business_id = np.load("data/yelp/business_id.npz", allow_pickle=True)
index_to_business = business_id["index_to_business"]
business_to_index = business_id["business_to_index"].item()


# map full set indexes to subset indexes
full_to_sub = {}
i = 0
for n in users:
    full_to_sub[n] = i
    i += 1

review_list = np.load("data/yelp/review.npy", allow_pickle=True)
cat_dist = [defaultdict(int) for _ in range(users.size)]
cat_set = set()
bus_to_user_dict = defaultdict(set)
for review in review_list:
    if review["user_id"] in user_id.keys() and review["business_id"] in business_to_index.keys():
        bus_idx = business_to_index[review["business_id"]]
        user_idx = full_to_sub[user_id[review["user_id"]]]
        bus_to_user_dict[review["business_id"]].add(user_idx)
        bus = business_list[bus_idx]
        '''if bus["categories"]:
            for cat in bus["categories"].split(", "):
                cat_dist[user_idx][cat] += 1
                cat_set.add(cat)'''
        cat_dist[user_idx][str(bus["stars"])]+=1
        cat_set.add(str(bus["stars"]))
print("Business to User dictionary through review")
print(list(bus_to_user_dict.items())[:5])

count = 0
for review in review_list:
    if review["business_id"] in business_to_index.keys():
        count+=1
print("number of edgesRB",count)



print("Category distribution list")
print(cat_dist[:5])
#print(cat_dist.shape)
tip_list = np.load("data/yelp/tip.npy", allow_pickle=True)
#cat_dist = [defaultdict(int) for _ in range(users.size)]
#cat_set = set()
bus_to_user_dict_tip = defaultdict(set)
for tip in tip_list:
    if tip["user_id"] in user_id.keys() and tip["business_id"] in business_to_index.keys():
        bus_idx = business_to_index[tip["business_id"]]
        user_idx = full_to_sub[user_id[tip["user_id"]]]
        bus_to_user_dict_tip[tip["business_id"]].add(user_idx)
print("Business to User dictionary through tip")
print(list(bus_to_user_dict_tip.items())[:5])

count = 0
for tip in tip_list:
    if tip["user_id"] in user_id.keys():
        count+=1
print("number of edgesUT",count)

edge_index = [[], []]
for user_set in bus_to_user_dict.values():
    user_set = list(user_set)
    for i in range(len(user_set) - 1):
        for j in range(i + 1, len(user_set)):
            edge_index[0].append(user_set[i])
            edge_index[1].append(user_set[j])
# fix edge_index by removing repeats and self edges
print("edge index pre tensor transformation")
print(edge_index)
edge_index = torch.tensor(edge_index)
print("edge index as tensor")
print(edge_index)
adj = geo.utils.to_dense_adj(edge_index)
adj = adj.squeeze()
print("Adjacency Matrix")
print(adj)
num_nodes = adj.size(dim=0)
adj[adj != 0] = 1
adj = torch.sub(adj, torch.eye(num_nodes))
print("adjacency matrix post normalization")
print(adj)
edge_index = geo.utils.dense_to_sparse(adj)[0]

edge_index_tip = [[], []]
for user_set in bus_to_user_dict_tip.values():
    user_set = list(user_set)
    for i in range(len(user_set) - 1):
        for j in range(i + 1, len(user_set)):
            edge_index_tip[0].append(user_set[i])
            edge_index_tip[1].append(user_set[j])
# fix edge_index by removing repeats and self edges
print("edge index pre tensor transformation")
print(edge_index_tip)
edge_index_tip = torch.tensor(edge_index_tip)
print("edge index as tensor")
print(edge_index_tip)
adj2 = geo.utils.to_dense_adj(edge_index_tip)
adj2 = adj2.squeeze()
print("Adjacency Matrix")
print(adj2)
num_nodes2 = adj2.size(dim=0)
adj2[adj2 != 0] = 1
adj2 = torch.sub(adj2, torch.eye(num_nodes2))
print("adjacency matrix post normalization")
print(adj2)
edge_index_tip = geo.utils.dense_to_sparse(adj2)[0]




cat_list = list(cat_set)
cat_to_index = {}
i = 0
for cat in cat_list:
    cat_to_index[cat] = i
    i += 1

cat_dist_final = np.zeros((users.size, len(cat_list)))
for i in range(users.size):
    for k, v in cat_dist[i].items():
        cat_dist_final[i][cat_to_index[k]] = v
    tot = np.sum(cat_dist_final[i, :])
    cat_dist_final[i, :] = cat_dist_final[i, :] / tot
    if i < 3:
        print(cat_dist_final[i, :])
print(cat_dist_final.shape)
print(user_feat.shape)
#np.savez("data/yelp/yelp_subset_graph", x=user_feat, y=cat_dist_final, edge_index=edge_index,edge_index2=edge_index_tip)


