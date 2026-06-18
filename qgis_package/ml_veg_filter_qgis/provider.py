# -*- coding: utf-8 -*-
import os
from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from .algorithms.train_algorithm import MLVegTrainAlgorithm
from .algorithms.reclass_algorithm import MLVegReclassAlgorithm
from .algorithms.filter_algorithm import MLVegFilterAlgorithm


class MLVegFilterProvider(QgsProcessingProvider):
    """Exposes all ML Vegetation Filter algorithms to QGIS Processing."""

    def id(self):
        return 'mlvegfilter'

    def name(self):
        return 'ML Vegetation Filter'

    def longName(self):
        return 'ML Vegetation Filter (TensorFlow MLP)'

    def icon(self):
        icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'icon.png')
        return QIcon(icon_path)

    def loadAlgorithms(self):
        self.addAlgorithm(MLVegTrainAlgorithm())
        self.addAlgorithm(MLVegReclassAlgorithm())
        self.addAlgorithm(MLVegFilterAlgorithm())
