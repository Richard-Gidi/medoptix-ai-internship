import numpy as np
import pandas as pd
from src.segmentation import fit_segmenter, evaluate_clustering

def test_segmentation_pipeline():
    # Generate dummy data
    X = pd.DataFrame(np.random.rand(100, 5))
    pipeline = fit_segmenter(X, n_clusters=3)
    labels = pipeline.named_steps['kmeans'].labels_
    score = evaluate_clustering(X, labels)
    assert score > 0.3, f'Silhouette score too low: {score}'
    assert not np.isnan(labels).any(), 'NaN cluster assignments found' 