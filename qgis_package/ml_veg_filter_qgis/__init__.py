# -*- coding: utf-8 -*-
"""
ML Vegetation Filter – QGIS Plugin
===================================
Segment vegetation from bare-Earth in dense RGB point clouds using TensorFlow.

Based on:
    Wernette, P.A. (2024). Segmenting Vegetation from bare-Earth in High-relief
    and Dense Point Clouds using Machine Learning.
    DOI: https://doi.org/10.5281/zenodo.10966854
"""


def classFactory(iface):
    from .plugin import MLVegFilterPlugin
    return MLVegFilterPlugin(iface)
