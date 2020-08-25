import numpy as np

w_bifpns = [64, 88, 112, 160, 224, 288, 384]
d_bifpns = [3, 4, 5, 6, 7, 7, 8]
d_heads = [3, 3, 3, 4, 4, 4, 5]
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]


_FEATURE_EXTRACTION_LAYERS = [
    ('block1a_project_bn','block2b_add','block3b_add','block5c_add','block7a_project_bn'),
    ('block1b_add','block2c_add', 'block3c_add', 'block5d_add', 'block7b_add'),
    ('block1b_add','block2c_add', 'block3c_add', 'block5d_add', 'block7b_add'),
    ('block1b_add','block2c_add', 'block3c_add', 'block5e_add', 'block7b_add'),
    ('block1b_add','block2d_add','block3d_add','block5f_add','block7b_add'),
    ('block1c_add','block2e_add','block3e_add','block5g_add','block7c_add'),
    ('block1c_add','block2f_add','block3f_add','block5h_add','block7c_add')
]

class Config(object):
    """docstring for Config"""
    def __init__(self, phi, weighted_bifpn=False, num_anchors=9):

        assert isinstance(weighted_bifpn,bool), 'weighted_bifpn should be a boolean'
        assert phi in range(7), f"phi should be between 0 and 7, not {phi}"
        self.phi = phi
        self.height = image_sizes[phi]
        self.width = image_sizes[phi]
        self.d_bifpn = d_bifpns[phi]
        self.w_bifpn = w_bifpns[phi]
        self.d_head = d_heads[phi]

        self.input_shape = (self.height, self.width, 3)

        self.weighted_bifpn = weighted_bifpn

        self.feature_extraction_layer_names = _FEATURE_EXTRACTION_LAYERS[phi]

        self.num_anchors = num_anchors


def B0Config(weighted_bifpn=False, num_anchors=9):
    return Config(0, weighted_bifpn=False, num_anchors=9)

def B1Config(weighted_bifpn=False, num_anchors=9):
    return Config(1, weighted_bifpn=False, num_anchors=9)

def B2Config(weighted_bifpn=False, num_anchors=9):
    return Config(2, weighted_bifpn=False, num_anchors=9)

def B3Config(weighted_bifpn=False, num_anchors=9):
    return Config(3, weighted_bifpn=False, num_anchors=9)

def B4Config(weighted_bifpn=False, num_anchors=9):
    return Config(4, weighted_bifpn=False, num_anchors=9)

def B5Config(weighted_bifpn=False, num_anchors=9):
    return Config(5, weighted_bifpn=False, num_anchors=9)

def B6Config(weighted_bifpn=False, num_anchors=9):
    return Config(6, weighted_bifpn=False, num_anchors=9)
