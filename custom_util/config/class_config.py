import enum

#CATEGORY_INFO
class Category(enum.Enum):
    FLAT = 0
    CONSTRUCTION = 1
    NONTARGET = 2
    OBJECT = 3


class_info = {
    # class                # id     # category_id                                 # color              # trainable
    'void':                {'id': 0, 'category_id': Category.CONSTRUCTION.value,   'color': [63, 63, 63],    'trainable': False},
    'runway':              {'id': 1, 'category_id': Category.FLAT.value,           'color': [0, 102, 0],     'trainable': True},
    'road':                {'id': 2, 'category_id': Category.FLAT.value,           'color': [128, 64, 128],  'trainable': True},
    'curb':                {'id': 3, 'category_id': Category.FLAT.value,           'color': [102, 102, 255], 'trainable': True},
    'mat':                 {'id': 4, 'category_id': Category.FLAT.value,           'color': [0, 100, 150],   'trainable': True},
    'braille-block':       {'id': 5, 'category_id': Category.FLAT.value,           'color': [255, 255, 0],   'trainable': True},
    'cross-walk':          {'id': 6, 'category_id': Category.FLAT.value,           'color': [64, 255, 0],    'trainable': True},
    'bicycle-road':        {'id': 7, 'category_id': Category.FLAT.value,           'color': [204, 51, 204],  'trainable': True},
    'speed-bump':          {'id': 8, 'category_id': Category.FLAT.value,           'color': [153, 102, 0],   'trainable': True},
    'manhole':             {'id': 9, 'category_id': Category.FLAT.value,           'color': [255, 200, 255], 'trainable': True},
    'terrain':             {'id': 10, 'category_id': Category.FLAT.value,          'color': [152, 251, 152], 'trainable': True},
    'vegetation':          {'id': 11, 'category_id': Category.CONSTRUCTION.value,  'color': [107, 142, 35],  'trainable': True},
    'poll':                {'id': 12, 'category_id': Category.CONSTRUCTION.value,  'color': [250, 0, 150],   'trainable': True},
    'pedestrian':          {'id': 13, 'category_id': Category.NONTARGET.value,     'color': [220, 20, 60],   'trainable': True},
    'tree':                {'id': 14, 'category_id': Category.CONSTRUCTION.value,  'color': [155, 50, 0],    'trainable': True},
    'vehicle':             {'id': 15, 'category_id': Category.NONTARGET.value,     'color': [250, 200, 100], 'trainable': True},
    'sky':                 {'id': 16, 'category_id': Category.NONTARGET.value,     'color': [70, 130, 180],  'trainable': True},
    'pole':                {'id': 17, 'category_id': Category.CONSTRUCTION.value,  'color': [250, 0, 150],   'trainable': True},
    'pole-group':          {'id': 18, 'category_id': Category.CONSTRUCTION.value,  'color': [116, 27, 71],   'trainable': True},
    'traffic-sign':        {'id': 19, 'category_id': Category.CONSTRUCTION.value,  'color': [64, 0, 128],    'trainable': True},
    'traffic-light-front': {'id': 20, 'category_id': Category.CONSTRUCTION.value,  'color': [250, 170, 30],  'trainable': True},
    'traffic-light-back':  {'id': 21, 'category_id': Category.CONSTRUCTION.value,  'color': [153, 0, 255],   'trainable': True},
    'fence':               {'id': 22, 'category_id': Category.CONSTRUCTION.value,  'color': [190, 153, 153], 'trainable': True},
    'dynamic':             {'id': 23, 'category_id': Category.NONTARGET.value,     'color': [150, 100, 255], 'trainable': True},
    'static':              {'id': 24, 'category_id': Category.CONSTRUCTION.value,  'color': [0, 128, 128],   'trainable': True},
    'forbidden-area':      {'id': 25, 'category_id': Category.FLAT.value,          'color': [255, 0, 0],     'trainable': False},
}

class_remap = {
    'pattern-block': 'runway',
    'deck': 'runway',
    'drain': 'curb',
    'cement': 'road',
    'asphalt': 'road',
    'non-pattern-block': 'terrain',
    'sand': 'terrain',
    'grass': 'terrain',
    'pebble': 'terrain',
    'wall': 'static',
    'building': 'static',
    'animal': 'dynamic',
}

def get_class_info():
    return class_info

def get_class_remap():
    return class_remap