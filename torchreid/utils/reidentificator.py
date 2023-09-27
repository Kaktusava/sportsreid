from __future__ import absolute_import
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from torchreid.utils import (
    check_isfile, load_pretrained_weights, compute_model_complexity
)
from torchreid.models import build_model
from .feature_extractor import FeatureExtractor
from torchreid import metrics

class Reidentificator(object):
    """
    Class for reidentification 
    using featrue extractor.
    """

    def __init__(
        self,
        model_name='vit_b_16',
        model_path='',
        query = None,
        image_size=(224, 224),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        device='cuda',
        verbose=True
    ):

        self.model = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            device=device,
            image_size=image_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            pixel_norm=pixel_norm,
            verbose=verbose
        )
        self.query = None
        if query is not None:
            self.set_query(query)

    def __call__(self, input):
        """
        input: list of cropped images
        output: list of top3 predicted ids
        """
        input_features = self.model(input)
        distmat = metrics.compute_distance_matrix(self.query, input_features.cpu(), 'euclidean')
        distmat = distmat.numpy()
        top3 = distmat.argsort(axis=1)[:,:3]
        return top3

    def set_query(self, query):
        """
        input:
        query list of tupples cropped images ad ids  [(cropped_frame_1, id_1), (cropped_frame_2, id_2),...]
        output:
        writes query features sorted by query ids to self.query
        """
        query = sorted(query, key=lambda x: x[1])
        query_features = self.model([i[0] for i in query])
        self.query = query_features.cpu()
