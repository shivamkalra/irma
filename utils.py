import numpy as np
from xlrd import open_workbook
import itertools
from skimage.io import imread
import os
import glob
import random
import skimage.transform as transform
import skimage as ski
import error
from scipy.io import loadmat
from skimage.color import rgb2grey

_package_root_dir = os.path.dirname(os.path.abspath(__file__))

_error = error.code(os.path.join(_package_root_dir, 'data/codes_08.txt'))

__all__ = [
    'evaluate_error', 'load_data', 'imread_norm', 'select_n_per_category'
]

_test_roi_dict = None
_train_roi_dict = None


def _convert_to_dict(arr):
    dict_data = {}

    for el in arr:
        key = str(el[0][0].split('.')[0])
        data = map(int, el[1][0])
        dict_data[key] = data

    return dict_data


def crop_image(rimage, odim):
    """
    Inverse radon transform `iradon` does not give the original dimension
    after reconstruction. This function fixes that issue.
    """
    df = (np.array(rimage.shape) - np.array(odim)) / 2.
    df = np.matrix([np.floor(df), np.ceil(df)]).T * np.matrix([[1., 0],
                                                               [0, -1.]])
    df = df + np.matrix([[0., rimage.shape[0]], [0, rimage.shape[1]]])
    df = np.array(df, dtype='int')
    return rimage[df[0, 0]:df[0, 1], df[1, 0]:df[1, 1]]


def evaluate_error(irma_code1, irma_code2):
    """
    Evaluates the IRMA error using given IRMA 2008 codes

    Args:
     irma_code1 (str)
     irma_code2 (str)
    """
    return _error.evaluate(irma_code1, irma_code2)


def load_test_data(path, categories):
    test_img_path = os.path.join(path, '2009/Testing Data/')
    img_file_paths = np.asarray(glob.glob(test_img_path + '*.png'))
    img_keys = np.asarray([
        os.path.splitext(os.path.basename(file_path))[0]
        for file_path in img_file_paths
    ])

    srt_idx = np.argsort(img_keys)

    img_file_paths = img_file_paths[srt_idx]
    img_keys = img_keys[srt_idx]

    code_xls = os.path.join(
        path, '2009/IRMA Code Testing/ImageCLEFmed2009_test_codes.03.xls')
    wb = open_workbook(code_xls)

    test_codes = {}
    for s in wb.sheets():
        for row in range(1, s.nrows):
            values = []
            for col in range(s.ncols):
                values.append(s.cell(row, col).value)
            image_number = str(int(values[0]))
            test_codes[image_number] = values[1:]

    irma_08_codes = [
        get_category_index(categories, '08', test_codes[key][-1])
        for key in img_keys
    ]

    irma_05_codes = [
        get_category_index(categories, '05', test_codes[key][6])
        for key in img_keys
    ]

    irma_06_codes = [
        get_category_index(categories, '05', test_codes[key][8])
        for key in img_keys
    ]

    irma_07_codes = [
        get_category_index(categories, '05', test_codes[key][9])
        for key in img_keys
    ]

    return {
        'image_id': np.asarray(img_keys),
        'image_path': np.asarray(img_file_paths),
        'irma_05_code': np.asarray(irma_05_codes),
        'irma_06_code': np.asarray(irma_06_codes),
        'irma_07_code': np.asarray(irma_07_codes),
        'irma_08_code': np.asarray(irma_08_codes),
        'irma_code': np.asarray(irma_08_codes)
    }


def get_category_index(categories, key, category):
    selected_categories = categories[key]
    return np.where(selected_categories ==
                    category)[0][0] if category in selected_categories else -1


def load_train_data(path, categories):
    train_img_path = os.path.join(
        path, '2009/Training Data/ImageCLEFmed2009_train.02/')
    img_file_paths = np.asarray(glob.glob(train_img_path + '*.png'))
    img_keys = np.asarray([
        os.path.splitext(os.path.basename(file_path))[0]
        for file_path in img_file_paths
    ])

    srt_idx = np.argsort(img_keys)
    img_keys = img_keys[srt_idx]
    img_file_paths = img_file_paths[srt_idx]

    train_codes_path = os.path.join(
        path, '2009/IRMA Code Training/ImageCLEFmed2009_train_codes.02.csv')

    train_codes = {}
    with open(train_codes_path, 'r') as f:
        next(f)
        for line in f:
            values = line.split(';')
            train_codes[values[0].strip(
            )] = [value.strip() for value in values[1:]]

    irma_05_codes = [
        get_category_index(categories, '05', train_codes[key][2])
        for key in img_keys
    ]
    irma_06_codes = [
        get_category_index(categories, '06', train_codes[key][4])
        for key in img_keys
    ]
    irma_07_codes = [
        get_category_index(categories, '07', train_codes[key][5])
        for key in img_keys
    ]
    irma_08_codes = [
        get_category_index(categories, '08', train_codes[key][6])
        for key in img_keys
    ]

    return {
        'image_id': np.asarray(img_keys),
        'image_path': np.asarray(img_file_paths),
        'irma_05_code': np.asarray(irma_05_codes),
        'irma_06_code': np.asarray(irma_06_codes),
        'irma_07_code': np.asarray(irma_07_codes),
        'irma_08_code': np.asarray(irma_08_codes),
        'irma_code': np.asarray(irma_08_codes)
    }


def load_categories(path):
    category_dir_path = os.path.join(path, '2009/Catergories')

    cat_classes = ['05', '06', '07', '08']
    categories = {}

    for cat_class in cat_classes:
        cat_file_path = os.path.join(category_dir_path,
                                     cat_class + '-classes.txt')
        categories[cat_class] = np.loadtxt(
            cat_file_path, delimiter=';', dtype=np.str)[:, 1]

    return categories


def load_data(path='./IRMA'):
    """
    Load the IRMA dataset from the given root directory
    """
    categories = load_categories(path)
    return {
        'category': categories,
        'train_data': load_train_data(path, categories),
        'test_data': load_test_data(path, categories)
    }


def select_n_per_category(data,
                          n=10,
                          category_key='irma_08_code',
                          include_non_catgeorized=False):
    """
    Get n training data per category (2008 categories)
    ``data`` is either training or testing
    """

    reduced_data = []

    groups = []
    for key, group in itertools.groupby(
            sorted(
                enumerate(data[category_key]), key=lambda x: x[1]),
            lambda x: x[1]):
        if include_non_catgeorized or key != -1:
            ids = [k for k, _ in group]
            n_mx = n if len(ids) > 10 else len(ids)
            reduced_data.extend(random.sample(ids, n_mx))
            groups.extend([key] * n_mx)

    return np.asarray(groups), np.asarray(reduced_data)


def pad_to_square(a, pad_value=0):

    mdim = max(a.shape)
    padded = pad_value * np.ones((mdim, mdim), dtype=a.dtype)

    o1 = int((mdim - a.shape[0])/2.)
    o2 = int((mdim - a.shape[1])/2.)

    padded[o1:a.shape[0]+o1, o2:a.shape[1]+o2] = a
    return padded


def get_id_from_path(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    if '_' in file_name:
        return file_name.split('_')[0]
    return file_name


def imread_norm(path, shape=(128, 128), roi=False):
    """
    First square an image then resize it
    """
    im = imread(path)

    if im.ndim is not 2:
        im = rgb2grey(im)

    if roi:
        image_id = get_id_from_path(path)

        if image_id in _train_roi_dict:
            crop_data = _train_roi_dict[image_id]
        else:
            crop_data = _test_roi_dict[image_id]

        im = im[crop_data[1]:crop_data[1] + crop_data[3], crop_data[0]:
                crop_data[0] + crop_data[2]]

    im = pad_to_square(ski.util.img_as_float(im))
    return transform.resize(im, shape)


if _train_roi_dict is None:
    train_roi_mat = loadmat(
        os.path.join(_package_root_dir, 'data/train_roi_map.mat'))
    _train_roi_dict = _convert_to_dict(train_roi_mat['result'][0])

if _test_roi_dict is None:
    test_roi_mat = loadmat(
        os.path.join(_package_root_dir, 'data/test_roi_map.mat'))
    _test_roi_dict = _convert_to_dict(test_roi_mat['result'][0])
