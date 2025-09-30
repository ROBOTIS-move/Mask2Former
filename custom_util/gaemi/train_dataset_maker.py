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
        train_data_list, val_data_list = self._split_dataset()
        self._save_record_json(train_data_list, val_data_list)

    def _split_dataset(self):
        dataset_path = self.gaemi_config.get('dataset_path', '')
        service_area = self.gaemi_config.get('service_area', '')
        target_path = os.path.join(dataset_path, service_area)
        if not os.path.exists(target_path):
            raise FileNotFoundError(f'Target path not found: {target_path}')

        img_files = glob(os.path.join(target_path, 'images', '*.jpg'))
        shuffle(img_files)

        split_ratio = self.gaemi_config.get('split_ratio', {'train': 0.8, 'val': 0.2})
        train_cnt = int(len(img_files) * split_ratio['train'])

        train_data_list = []
        val_data_list = []
        train_data_list.extend(img_files[:train_cnt])
        val_data_list.extend(img_files[train_cnt:])

        return train_data_list, val_data_list

    def _save_record_json(self, train_data_list, val_data_list):
        record_path = self.gaemi_config.get('json_record_path', '')
        service_area = self.gaemi_config.get('service_area', '')

        if os.path.exists(record_path):
            existing_record = self._read_json(record_path)
            existing_record['train'][service_area] = train_data_list
            existing_record['val'][service_area] = val_data_list
            record = existing_record
        else:
            record = {'train': {service_area: train_data_list},
                      'val': {service_area: val_data_list}}

        self._write_json(record_path, record)

    def _read_json(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'JSON file not found at {path}')
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def _write_json(self, path, data):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        self.logger.info(f'Saved JSON record at {path}')


if __name__ == '__main__':
    maker = DatasetMaker()
    maker.execute()