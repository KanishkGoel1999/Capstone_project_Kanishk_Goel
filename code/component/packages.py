# Standard Library Imports
import os                          # For directory and file handling
import sys                         # For system-level operations
import time                        # To measure execution time
import yaml                        # For reading configuration files
import pickle                      # For saving/loading models
import argparse                    # For parsing command-line arguments
import warnings                    # For managing warnings
from datetime import datetime      # For working with date and time


# Progress Bar and Display
from tqdm import tqdm              # For showing progress bars
import prettytable                 # For printing tabular data in console


# Data Manipulation and Visualization
import numpy as np                 # For numerical operations
import pandas as pd                # For data manipulation
from matplotlib import pyplot as plt  # For plotting graphs


# Scikit-learn Utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, auc
)

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import xgboost as xgb
import category_encoders as ce     # For advanced categorical encodings

# PyTorch and PyTorch Geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_networkx

# DGL (Deep Graph Library)
import dgl
import dgl.function as fn
from dgl.nn import RelGraphConv

import networkx as nx              # For graph visualization and analysis
