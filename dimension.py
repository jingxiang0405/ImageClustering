from sklearn.manifold import TSNE


def reduce(features, tech="tsne", standardize=True):
    if tech == "tsne":
        tsne = TSNE(n_components=2, random_state=42)
        result = tsne.fit_transform(features)

        # Standardize
        if standardize:
            min_x, min_y = result.min(axis=0)
            max_x, max_y = result.max(axis=0)
            result = (result - [min_x, min_y]) / [max_x - min_x, max_y - min_y]
            result = result * 15

        return result
