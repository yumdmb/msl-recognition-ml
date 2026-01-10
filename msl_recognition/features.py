"""
Feature engineering and landmark preprocessing
"""
import numpy as np


def normalize_landmarks(landmarks):
    """
    Normalize landmarks to be centered at wrist and scaled by palm size.
    
    Args:
        landmarks: Array of shape (n_samples, 63) where each row is flattened x,y,z coords
    
    Returns:
        Normalized landmarks array of same shape
    """
    landmarks = landmarks.reshape(-1, 21, 3)
    
    # Center around wrist (landmark 0)
    centered = landmarks - landmarks[:, 0:1, :]
    
    # Scale by palm size (distance between wrist and middle finger MCP)
    scale = np.linalg.norm(centered[:, 9:10, :], axis=2)  # MCP joint
    normalized = centered / (scale + 1e-8)[:, :, np.newaxis]
    
    return normalized.reshape(-1, 63)
