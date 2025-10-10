import os
import sys
import yaml
import json
import logging
import numpy as np
import cv2

from glob import glob
from random import shuffle
from PIL import Image
from tqdm import tqdm

# custom_util 경로를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
custom_util_path = os.path.dirname(current_dir)
if custom_util_path not in sys.path:
    sys.path.insert(0, custom_util_path)

from config.class_config import (
    get_class_info,
    get_class_remap
)


class DatasetMaker:

    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.gaemi_config = self._get_config(base_path)
        self.class_info = get_class_info()
        self.class_remap = get_class_remap()

        self.existing_train_data = []
        self.existing_val_data = []
        self.existing_test_data = []

        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s][%(levelname)s][%(name)s]: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _get_config(self, base_path):
        config_path = os.path.join(base_path, '../config/config.yaml')
        if not os.path.exists(config_path):
            self.logger.error(f'Config file not found at {config_path}')
            raise FileNotFoundError(f'Config file not found at {config_path}')

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        required_keys = ['gaemi_GT_data']
        for key in required_keys:
            if key not in config:
                raise KeyError(f'Missing required config key: {key}')
        return config['gaemi_GT_data']

    def execute(self):
        self.existing_train_data, self.existing_val_data, self.existing_test_data = \
            self._get_existing_data_lists()
        train_data_list, val_data_list, test_data_list = self._split_dataset()
        self._save_img_record_json(train_data_list, type='train')
        self._save_img_record_json(val_data_list, type='val')
        self._save_img_record_json(test_data_list, type='test')
        self._make_panoptic_annotations(train_data_list, type='train')
        self._make_panoptic_annotations(val_data_list, type='val')
        self._make_panoptic_annotations(test_data_list, type='test')

    def _get_existing_data_lists(self):
        record_path = self.gaemi_config.get('record_path', '')
        record_file_name = self.gaemi_config.get('record_file_name', 'record')
        service_area = self.gaemi_config.get('service_area', '')

        # Create file paths for each split
        train_file_path = os.path.join(record_path, f'{record_file_name}_train.json')
        val_file_path = os.path.join(record_path, f'{record_file_name}_val.json')
        test_file_path = os.path.join(record_path, f'{record_file_name}_test.json')

        existing_train = []
        existing_val = []
        existing_test = []

        # Read train data
        if os.path.exists(train_file_path):
            try:
                train_record = self._read_json(train_file_path)
                existing_train = train_record.get(service_area, [])
            except (json.JSONDecodeError, FileNotFoundError) as e:
                self.logger.warning(f'Failed to read train record: {e}')

        # Read val data
        if os.path.exists(val_file_path):
            try:
                val_record = self._read_json(val_file_path)
                existing_val = val_record.get(service_area, [])
            except (json.JSONDecodeError, FileNotFoundError) as e:
                self.logger.warning(f'Failed to read val record: {e}')

        # Read test data
        if os.path.exists(test_file_path):
            try:
                test_record = self._read_json(test_file_path)
                existing_test = test_record.get(service_area, [])
            except (json.JSONDecodeError, FileNotFoundError) as e:
                self.logger.warning(f'Failed to read test record: {e}')

        return existing_train, existing_val, existing_test

    def _split_dataset(self):
        dataset_path = self.gaemi_config.get('dataset_path', '')
        service_area = self.gaemi_config.get('service_area', '')
        target_path = os.path.join(dataset_path, service_area)
        if not os.path.exists(target_path):
            raise FileNotFoundError(f'Target path not found: {target_path}')

        # Bring all image files
        all_img_files = glob(os.path.join(target_path, 'images', '*.jpg'))
        split_ratio = self.gaemi_config.get('split_ratio', {'train': 0.7, 'val': 0.2, 'test': 0.1})

        # Get new images excluding existing ones
        existing_all_data = set(self.existing_train_data + self.existing_val_data + self.existing_test_data)
        new_img_files = [img for img in all_img_files if img not in existing_all_data]

        # Extract test data from new images (keep existing test unchanged)
        new_test_count = int(len(new_img_files) * split_ratio['test'])
        new_test_data = new_img_files[:new_test_count]
        shuffle(new_img_files)  # Shuffle new data only

        # Accumulate test data: existing + new
        final_test_data = self.existing_test_data + new_test_data

        # Make train/val data: exclude all test data from all images
        available_for_train_val = [img for img in all_img_files if img not in set(final_test_data)]
        shuffle(available_for_train_val)  # Shuffle train/val data

        # Recalculate train/val split ratio (excluding test)
        train_val_total = len(available_for_train_val)
        train_ratio_adjusted = split_ratio['train'] / (split_ratio['train'] + split_ratio['val'])
        train_count = int(train_val_total * train_ratio_adjusted)

        final_train_data = available_for_train_val[:train_count]
        final_val_data = available_for_train_val[train_count:]

        return final_train_data, final_val_data, final_test_data

    def _save_img_record_json(self, data_list, type='train'):
        record_path = self.gaemi_config.get('record_path', '')
        record_file_name = self.gaemi_config.get('record_file_name', 'record')
        service_area = self.gaemi_config.get('service_area', '')

        # Create directory if it doesn't exist
        if not os.path.exists(record_path):
            if not self._is_path_allowed(record_path):
                raise PermissionError(f'Access denied: Cannot create directory {record_path}')
            os.makedirs(record_path, exist_ok=True)

        # Create file paths for each split
        file_path = os.path.join(record_path, f'{record_file_name}_{type}.json')

        # Save type data
        record = {}
        if os.path.exists(file_path):
            try:
                record = self._read_json(file_path)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                self.logger.warning(f'Failed to read existing {type} record: {e}. Creating new record.')

        record[service_area] = data_list
        self._write_json(file_path, record)

    def _make_panoptic_annotations(self, data_list, type='train'):
        record_path = self.gaemi_config.get('record_path', '')
        gaemi_annotations = []
        for data_path in tqdm(data_list):
            # print(data_path)
            mask_annotation_path = data_path.replace('/images/', '/labels/').replace('.jpg', '.png')
            json_annotation_path = data_path.replace('/images/', '/labels/').replace('.jpg', '.json')
            if not os.path.exists(mask_annotation_path):
                self.logger.warning(f'Annotation file not found for image {data_path}')
                continue

            annotation_img = Image.open(mask_annotation_path)
            annotation_np = np.array(annotation_img)

            record = {
                'file_name': data_path,
                'image_id': os.path.basename(data_path).replace('.jpg', ''),
                'height': annotation_img.height,
                'width': annotation_img.width,
                'annotations': []
            }

            # Process the annotation to create panoptic format
            panoptic_annotation = self._calculate_annotation(annotation_np, json_annotation_path)
            record['annotations'] = panoptic_annotation
            gaemi_annotations.append(record)

        save_path = os.path.join(record_path, f'panoptic_{type}_annotations.json')
        self._write_json(save_path, gaemi_annotations)

    def _calculate_annotation(self, annotation_np, json_path):
        """JSON annotation 파일에서 각 segment의 area와 bbox를 계산"""
        json_info = self._read_json(json_path)
        image_height = json_info.get('imageHeight', annotation_np.shape[0])
        image_width = json_info.get('imageWidth', annotation_np.shape[1])
        
        annotations = []
        segment_id = 1  # Start from 1

        for shp in json_info['shapes']:
            label = shp['label']
            
            # Convert label if in remap (e.g., 'cement' -> 'road')
            converted_label = self.class_remap.get(label, label)
            
            # Get label info from class_info
            # print(self.class_info)
            # print(type(converted_label))
            try:
                label_info = self.class_info[converted_label]
            except KeyError:
                self.logger.warning(f'Label "{label}" (converted: "{converted_label}") not found in class_info. Skipping.')
                continue
            
            label_id = label_info.get('id', 0)
            category_id = label_info.get('category_id', 0)
            
            if label_id == 0:
                self.logger.warning(f'Label "{label}" has id=0. Skipping.')
                continue

            # Get polygon points
            polygon = shp['points']
            
            # Calculate area and bbox
            area, bbox = self._calculate_area_and_bbox(polygon, image_height, image_width)
            
            # iscrowd는 instance segmentation에서만 사용, panoptic에서는 기본 0
            iscrowd = 0

            annotation_info = {
                'label': label,
                "id": segment_id,
                "category_id": int(category_id),
                "area": int(area),
                "bbox": bbox,
                "iscrowd": iscrowd
            }
            
            annotations.append(annotation_info)
            segment_id += 1

        return annotations

    def _calculate_area_and_bbox(self, polygon, image_height, image_width):
        """Polygon 좌표로부터 area와 bbox를 계산"""
        # Convert polygon points to numpy array
        polygon_array = np.array(polygon, dtype=np.int32)
        
        # Create binary mask
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_array], 1)
        
        # Calculate area (number of pixels in the mask)
        area = np.sum(mask)
        
        # Calculate bounding box
        # Horizontal projection
        hor = np.sum(mask, axis=0)
        hor_idx = np.nonzero(hor)[0]
        
        if len(hor_idx) == 0:
            # Empty mask
            return 0, [0, 0, 0, 0]
        
        x = hor_idx[0]
        width = hor_idx[-1] - x + 1
        
        # Vertical projection
        vert = np.sum(mask, axis=1)
        vert_idx = np.nonzero(vert)[0]
        y = vert_idx[0]
        height = vert_idx[-1] - y + 1
        
        bbox = [int(x), int(y), int(width), int(height)]
        
        return area, bbox


    def _read_json(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f'JSON file not found at {path}')
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f'Invalid JSON format in {path}: {e}')

    def _write_json(self, path, data):
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            self.logger.info(f'Successfully saved JSON record at {path}')
        except (IOError, OSError) as e:
            self.logger.error(f'Failed to write JSON file at {path}: {e}')
            raise


if __name__ == "__main__":
    maker = DatasetMaker()
    maker.execute()