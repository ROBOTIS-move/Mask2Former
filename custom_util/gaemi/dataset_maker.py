import os
import yaml
import json
import logging

from glob import glob
from random import shuffle

from Mask2Former.custom_util.config.class_config import (
    get_class_info,
    get_class_remap
)
from Mask2Former.custom_util.gaemi.panoptic_util import PanopticUtil
from Mask2Former.custom_util.gaemi.semantic_util import SemanticUtil


class DatasetMaker:

    def __init__(self, data_type='DrivingAreaSegmentation', convert_type='semantic'):
        self.data_type = data_type
        self.convert_type = convert_type
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.gaemi_config = self._get_config(base_path)
        self.class_info = get_class_info()
        self.class_remap = get_class_remap()

        self.panoptic = PanopticUtil(
            self.gaemi_config,
            self.class_info,
            self.class_remap
        )
        self.semantic = SemanticUtil(
            self.data_type,
            self.gaemi_config,
            self.class_info,
            self.class_remap
        )

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
        self._save_img_record_json(train_data_list, target_type='train')
        self._save_img_record_json(val_data_list, target_type='val')
        self._save_img_record_json(test_data_list, target_type='test')

        if self.convert_type == 'panoptic':
            json_path, annotations = self.panoptic.make_panoptic_annotations(
                self._get_target_file_path('train'),
                target_type='train'
            )
            self._write_json(json_path, annotations)
            json_path, annotations = self.panoptic.make_panoptic_annotations(
                self._get_target_file_path('val'),
                target_type='val'
            )
            self._write_json(json_path, annotations)
            json_path, annotations = self.panoptic.make_panoptic_annotations(
                self._get_target_file_path('test'),
                target_type='test'
            )
            self._write_json(json_path, annotations)

            # Create panoptic PNG files
            self.panoptic.create_panoptic_png(
                target_type='train'
            )
            self.panoptic.create_panoptic_png(
                target_type='val'
            )
            self.panoptic.create_panoptic_png(
                target_type='test'
            )

        elif self.convert_type == 'semantic':
            # Train set
            train_json_path, train_annotations = self.semantic.create_semantic_annotations(
                self._get_target_file_path('train'),
                target_type='train'
            )
            self._write_json(train_json_path, train_annotations)
            self.semantic.create_semantic_png(
                train_annotations,
                target_type='train'
            )
            # Val set
            val_json_path, val_annotations = self.semantic.create_semantic_annotations(
                self._get_target_file_path('val'),
                target_type='val'
            )
            self._write_json(val_json_path, val_annotations)
            self.semantic.create_semantic_png(
                val_annotations,
                target_type='val'
            )
            # Test set
            test_json_path, test_annotations = self.semantic.create_semantic_annotations(
                self._get_target_file_path('test'),
                target_type='test'
            )
            self._write_json(test_json_path, test_annotations)
            self.semantic.create_semantic_png(
                test_annotations,
                target_type='test'
            )


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

    def _save_img_record_json(self, data_list, target_type='train'):
        record_path = self.gaemi_config.get('record_path', '')
        record_file_name = self.gaemi_config.get('record_file_name', 'record')
        service_area = self.gaemi_config.get('service_area', '')

        # Create directory if it doesn't exist
        os.makedirs(record_path, exist_ok=True)

        # Create file paths for each split
        file_path = os.path.join(record_path, f'{record_file_name}_{target_type}.json')

        # Save type data
        record = {}
        if os.path.exists(file_path):
            try:
                record = self._read_json(file_path)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                self.logger.warning(f'Failed to read existing {target_type} record: {e}. Creating new record.')

        record[service_area] = data_list
        self._write_json(file_path, record)

    def _get_target_file_path(self, target_type='train'):
        record_path = self.gaemi_config.get('record_path', '')
        record_file_name = self.gaemi_config.get('record_file_name', '')

        target_file = os.path.join(record_path, f"{record_file_name}_{target_type}.json")
        target_info = self._read_json(target_file)

        return_list = []
        for service_area in target_info:
            return_list += target_info[service_area]

        return return_list

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