import os
import yaml
import json
import logging

from glob import glob
from random import shuffle


class DatasetMaker:

    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.gaemi_config = self._get_config(base_path)
        self.existing_train_data = []
        self.existing_val_data = []
        self.existing_test_data = []

        # Set base paths for security checks
        self.allowed_base_paths = [
            os.path.dirname(base_path),
            os.path.dirname(os.path.dirname(base_path)),
            self.gaemi_config.get('dataset_path', ''),
        ]

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
        self._save_record_json(train_data_list, val_data_list, test_data_list)

    def _get_existing_data_lists(self):
        record_path = self.gaemi_config.get('json_record_path', '')
        service_area = self.gaemi_config.get('service_area', '')

        if os.path.exists(record_path):
            record = self._read_json(record_path)
            existing_train = record.get('train', {}).get(service_area, [])
            existing_val = record.get('val', {}).get(service_area, [])
            existing_test = record.get('test', {}).get(service_area, [])
            return existing_train, existing_val, existing_test
        return [], [], []

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

        # 디버그 정보 출력
        # print(f'Total images: {len(all_img_files)}')
        # print(f'Existing data - Train: {len(self.existing_train_data)}, '
        #       f'Val: {len(self.existing_val_data)}, Test: {len(self.existing_test_data)}')
        # print(f'New images: {len(new_img_files)}, New test: {len(new_test_data)}')
        # print(f'Final split - Train: {len(final_train_data)}, '
        #       f'Val: {len(final_val_data)}, Test: {len(final_test_data)}')

        return final_train_data, final_val_data, final_test_data

    def _is_path_allowed(self, path):
        try:
            real_path = os.path.realpath(path)
            for allowed_path in self.allowed_base_paths:
                if allowed_path and real_path.startswith(os.path.realpath(allowed_path)):
                    return True
            return False
        except (OSError, ValueError):
            return False

    def _save_record_json(self, train_data_list, val_data_list, test_data_list):
        record_path = self.gaemi_config.get('json_record_path', '')
        service_area = self.gaemi_config.get('service_area', '')

        # Check if record path is within allowed directories
        if not self._is_path_allowed(record_path):
            raise PermissionError(f'Access denied: {record_path} is outside allowed directories')

        # Check if record file exists
        if os.path.exists(record_path):
            try:
                record = self._read_json(record_path)
                # if keys missing, initialize them
                for key in ['train', 'val', 'test']:
                    if key not in record:
                        record[key] = {}
            except (json.JSONDecodeError, FileNotFoundError) as e:
                self.logger.warning(f'Failed to read existing record: {e}. Creating new record.')
                record = {'train': {}, 'val': {}, 'test': {}}
        else:
            # If directory doesn't exist, create it
            record_dir = os.path.dirname(record_path)
            if not self._is_path_allowed(record_dir):
                raise PermissionError(f'Access denied: Cannot create directory {record_dir}')
            os.makedirs(record_dir, exist_ok=True)
            record = {'train': {}, 'val': {}, 'test': {}}

        # Update data by service area
        record['train'][service_area] = train_data_list
        record['val'][service_area] = val_data_list
        record['test'][service_area] = test_data_list

        # debug
        # print(f'Saving record with Train: {len(train_data_list)},'
        #       f' Val: {len(val_data_list)},'
        #       f' Test: {len(test_data_list)}'
        #     )

        self._write_json(record_path, record)

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