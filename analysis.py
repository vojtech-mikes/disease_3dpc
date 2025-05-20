import pathlib
from sklearn.metrics.pairwise import cosine_similarity
import stats

# Load the plants paths
data_path = pathlib.Path("./data/").absolute()

ill_feat_vectors = stats.create_feature_vectors(data_path, "ill")

ok_feat_vectors = stats.create_feature_vectors(data_path, "ok")

ill_cosine_sim_matrix = cosine_similarity(ill_feat_vectors)

ok_cosine_sim_matrix = cosine_similarity(ok_feat_vectors)

all_feat_vectors = ill_feat_vectors + ok_feat_vectors

all_cosine_sim_matrix = cosine_similarity(all_feat_vectors)


