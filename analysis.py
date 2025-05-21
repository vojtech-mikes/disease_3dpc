import pathlib
from sklearn.metrics.pairwise import cosine_similarity
import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the plants paths
data_path = pathlib.Path("./data/").absolute()

ill_feat_vectors = stats.create_feature_vectors(data_path, "ill")

ok_feat_vectors = stats.create_feature_vectors(data_path, "ok")

ill_cossim = cosine_similarity(ill_feat_vectors)
ok_cossim = cosine_similarity(ok_feat_vectors)

all_dataframe = pd.concat([ill_feat_vectors, ok_feat_vectors], ignore_index=False)

all_cossim_df = pd.DataFrame(cosine_similarity(all_dataframe), index=all_dataframe.index, columns=all_dataframe.index)

# Plot the heatmap
plt.figure(figsize=(14, 12))  # adjust size as needed
sns.heatmap(
    all_cossim_df,
    annot=True,
    fmt=".4f",
    cmap="coolwarm",
    xticklabels=True,
    yticklabels=True,
    annot_kws={"size": 6}  # adjust font size for clarity
)

plt.xticks(rotation=45, ha='right')
plt.title("Cosine Similarity Between Samples")
plt.tight_layout()
plt.savefig("plot.png")
