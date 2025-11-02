import numpy as np

def match_features(target_features: np.ndarray, frame_features: np.ndarray) -> float:
    similarity = np.dot(target_features, frame_features)
    return float(similarity)
