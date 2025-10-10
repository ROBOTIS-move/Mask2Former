import enum

#CATEGORY_INFO
class Category(enum.Enum):
    FLAT = 0
    CONSTRUCTION = 1
    NONTARGET = 2
    OBJECT = 3


class_info = {
    # class                # id     # category_id                                 # color
    'runway':              {'id': 1, 'category_id': Category.FLAT.value,           'color': [0, 102, 0]},
    'road':                {'id': 2, 'category_id': Category.FLAT.value,           'color': [128, 64, 128]},
    'curb':                {'id': 3, 'category_id': Category.FLAT.value,           'color': [102, 102, 255]},
    'mat':                 {'id': 4, 'category_id': Category.FLAT.value,           'color': [0, 100, 150]},
    'braille-block':       {'id': 5, 'category_id': Category.FLAT.value,           'color': [255, 255, 0]},
    'cross-walk':          {'id': 6, 'category_id': Category.FLAT.value,           'color': [64, 255, 0]},
    'bicycle-road':        {'id': 7, 'category_id': Category.FLAT.value,           'color': [204, 51, 204]},
    'speed-bump':          {'id': 8, 'category_id': Category.FLAT.value,           'color': [153, 102, 0]},
    'manhole':             {'id': 9, 'category_id': Category.FLAT.value,           'color': [255, 200, 255]},
    'terrain':             {'id': 10, 'category_id': Category.FLAT.value,          'color': [152, 251, 152]},
    'vegetation':          {'id': 11, 'category_id': Category.CONSTRUCTION.value,  'color': [107, 142, 35]},
    'poll':                {'id': 12, 'category_id': Category.CONSTRUCTION.value,  'color': [250, 0, 150]},
    'pedestrian':          {'id': 13, 'category_id': Category.NONTARGET.value,     'color': [220, 20, 60]},
    'tree':                {'id': 14, 'category_id': Category.CONSTRUCTION.value,  'color': [155, 50, 0]},
    'vehicle':             {'id': 15, 'category_id': Category.NONTARGET.value,     'color': [250, 200, 100]},
    'sky':                 {'id': 16, 'category_id': Category.NONTARGET.value,     'color': [70, 130, 180]},
    'pole':                {'id': 17, 'category_id': Category.CONSTRUCTION.value,  'color': [250, 0, 150]},
    'pole-group':          {'id': 18, 'category_id': Category.CONSTRUCTION.value,  'color': [116, 27, 71]},
    'traffic-sign':        {'id': 19, 'category_id': Category.CONSTRUCTION.value,  'color': [64, 0, 128]},
    'traffic-light-front': {'id': 20, 'category_id': Category.CONSTRUCTION.value,  'color': [250, 170, 30]},
    'traffic-light-back':  {'id': 21, 'category_id': Category.CONSTRUCTION.value,  'color': [153, 0, 255]},
    'fence':               {'id': 22, 'category_id': Category.CONSTRUCTION.value,  'color': [190, 153, 153]},
    'dynamic':             {'id': 23, 'category_id': Category.NONTARGET.value,     'color': [150, 100, 255]},
    'static':              {'id': 24, 'category_id': Category.CONSTRUCTION.value,  'color': [0, 128, 128]},
    'forbidden-area':      {'id': 25, 'category_id': Category.FLAT.value,          'color': [255, 0, 0]},
    'void':                {'id': 26, 'category_id': Category.CONSTRUCTION.value,  'color': [63, 63, 63]},
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