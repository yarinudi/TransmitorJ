# %%
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA


# %%
df = pd.read_csv(r'C:\Users\yarin\University\PhD\GaitLab\MyWorks\678f38936322f76869de2083\analysis\amit_feats\amit_WHS_all_features_preprocessed.csv')
df = df.round(3)
df = df.set_index('Subject')
df.head()

# %% NaNs exploration
vals = df.isna().sum().sort_values(ascending=False)
vals = vals[vals>0]
fig = px.bar(vals, title="Number of NaNs per feature")
fig.show(renderer="browser")

# %% Display correlation matrix of the features
corr_matrix = df.corr()
corr_matrix

heatmap = px.imshow(corr_matrix)
heatmap.show(renderer="browser")

# %% PCA dimension reduction 
pca = PCA(n_components=2)
embeddings = pca.fit_transform(df)
print(pca.explained_variance_ratio_)

# Display the embeddings
fig_pca = px.scatter(embeddings, x=0, y=1, title="PCA of Amit's features")
fig_pca.show(renderer="browser")

# %%
