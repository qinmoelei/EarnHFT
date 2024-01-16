import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# X=np.load("RL/tree_importance/X.npy")
# y=np.load("RL/tree_importance/y.npy")
# print(X.shape)
# print(y.shape)
# dt = DecisionTreeClassifier(random_state=42)
# dt.fit(X, y)
# features=np.load("data/selected_features.npy").tolist()
# features.append("previous_action")
# importance = dt.feature_importances_
# feature_importance=dict()
# for i,v in enumerate(importance):
#     if v!=0:
#         print(features[i])
#         feature_importance[features[i]]=v
# np.save("RL/tree_importance/feature_importance.npy",feature_importance)




X=np.load("RL/tree_importance/X_L.npy")
y=np.load("RL/tree_importance/y_L.npy")
print(X.shape)
print(y.shape)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X, y)
features=np.load("data/selected_features.npy").tolist()
features.append("previous_action")
features.append("holding_length")

importance = dt.feature_importances_
feature_importance=dict()
for i,v in enumerate(importance):
    if v!=0:
        print(features[i],v)
        feature_importance[features[i]]=v
np.save("RL/tree_importance/feature_importance_L.npy",feature_importance)










































