"""
I/O helper utilities for managing experiment outputs.

Provides functions to generate unique run tags, create output directories,
and save trained models, hyperparameters, and normalization constants.
"""

import os
import json
import uuid
import torch


def make_tag() -> str:
    """
    Generate a short, random hexadecimal string for tagging experiment runs.
    
    Returns:
        A unique 8-character string.
    """
    return uuid.uuid4().hex[:8]


def create_output_dir(tag: str):
    """
    Create a directory named by the given tag, if it does not already exist.
    
    Args:
        tag: The name of the directory to create.
    """
    os.makedirs(tag, exist_ok=True)


def save_models(models: list, tag: str, prefix: str = None):
    """
    Save each PyTorch model’s state dict into the output directory.
    
    Args:
        models: List of PyTorch nn.Module instances to save.
        tag:    Output directory name (created via create_output_dir).
        prefix: Filename prefix; defaults to the tag if not provided.
        
    Files written:
        {tag}/{prefix}_fold1.pth, {tag}/{prefix}_fold2.pth, …
    """
    prefix = prefix or tag
    for i, model in enumerate(models, start=1):
        path = os.path.join(tag, f"{prefix}_fold{i}.pth")
        torch.save(model.state_dict(), path)


def save_hyperparams(params: dict, tag: str, filename: str = None):
    """
    Dump training hyperparameters to a JSON file in the output directory.
    
    Args:
        params:   Dictionary of hyperparameters.
        tag:      Output directory name.
        filename: JSON filename; defaults to "{tag}_hyperparams.json".
    """
    fn = filename or f"{tag}_hyperparams.json"
    with open(os.path.join(tag, fn), 'w') as f:
        json.dump(params, f, indent=4)


def save_norms(norms: dict, tag: str, filename: str = None):
    """
    Dump normalization constants (mean/std for targets and features) to JSON.
    
    Args:
        norms:    Dictionary with keys 'y_mean','y_std','feat_mean','feat_std'.
        tag:      Output directory name.
        filename: JSON filename; defaults to "{tag}_norms.json".
    """
    fn = filename or f"{tag}_norms.json"
    # Convert all values to native Python floats for JSON serialization
    safe = {
        'y_mean':   float(norms['y_mean']),
        'y_std':    float(norms['y_std']),
        'feat_mean': {k: float(v) for k, v in norms['feat_mean'].items()},
        'feat_std':  {k: float(v) for k, v in norms['feat_std'].items()},
    }
    with open(os.path.join(tag, fn), 'w') as f:
        json.dump(safe, f, indent=4)
