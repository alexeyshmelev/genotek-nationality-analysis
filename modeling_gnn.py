import time
import torch
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import torch.nn as nn
from sklearn import metrics
from numba import njit, prange
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import InMemoryDataset, Data
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


class DatasetGenerator():
    def __init__(self, path, train_size):
        self.train_size = train_size
        self.df = pd.read_csv(path)
        
    def get_unique_node_classes(df):
        return pd.concat([df['label_id1'], df['label_id2']], axis=0).unique().tolist()
        
    def geterate_train_and_test_nodes():
        unique_node_classes = get_unique_node_classes(self.df)