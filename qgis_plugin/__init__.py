def classFactory(iface):
    from .mlp_veg_seg import MlpVegSeg
    return MlpVegSeg(iface)