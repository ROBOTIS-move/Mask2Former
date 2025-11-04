import os
import json
import logging
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm

class SemanticUtil:

    def __init__(self, data_type, config, class_info, class_remap):
        self.data_type = data_type
        self.gaemi_config = config
        self.class_info = class_info
        self.class_remap = class_remap

        self._set_logger()

        # Create dataset_id to contiguous_id mapping for trainable classes only
        self.dataset_id_to_contiguous_id = self._create_id_mapping()

    def _create_id_mapping(self):
        trainable_classes = [
            (name, info['id'])
            for name, info in self.class_info.items()
            if info.get('trainable', True)
        ]
        # Sort by dataset_id to ensure consistent mapping
        trainable_classes.sort(key=lambda x: x[1])

        dataset_id_to_contiguous_id = {}
        contiguous_id = 1  # Start from 1 (0 is reserved for background)

        for _, dataset_id in trainable_classes:
            dataset_id_to_contiguous_id[dataset_id] = contiguous_id
            contiguous_id += 1

        self.logger.info(f"Created ID mapping for {len(dataset_id_to_contiguous_id)} trainable classes")
        self.logger.info(f"  contiguous_id range: 1 to {contiguous_id - 1}")
        self.logger.debug(f"  Mapping: {dataset_id_to_contiguous_id}")

        return dataset_id_to_contiguous_id

    def _set_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s][%(levelname)s][%(name)s]: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info('SemanticUtil initialized')

    def create_semantic_annotations(self, data_list, target_type='train'):
        self.logger.info(f"Creating semantic annotation JSON for {target_type} set...")

        record_path = self.gaemi_config.get('record_path', '')
        model_train_data_path = self.gaemi_config.get('model_train_data_path', '')

        gaemi_annotations = []
        mask_missing_count = 0
        json_missing_count = 0

        for data_path in tqdm(data_list):
            color_mask_img_path = data_path.replace('/images/', '/labels/').replace('.jpg', '.png')
            json_annotation_path = data_path.replace('/images/', '/labels/').replace('.jpg', '.json')
            train_mask_img_path = os.path.join(
                model_train_data_path,
                self.data_type,
                target_type,
                os.path.basename(color_mask_img_path))

            if not os.path.exists(color_mask_img_path):
                self.logger.warning(f"Mask annotation not found: {color_mask_img_path}")
                mask_missing_count += 1
                continue

            if not os.path.exists(json_annotation_path):
                self.logger.warning(f"JSON annotation not found: {json_annotation_path}")
                json_missing_count += 1
                continue

            color_mask_img = Image.open(color_mask_img_path)

            record = {
                "file_name": data_path,
                "image_id": os.path.basename(data_path).replace('.jpg', ''),
                "height": color_mask_img.height,
                "width": color_mask_img.width,
                "sem_seg_file_name": train_mask_img_path,
                "json_file_name": json_annotation_path
            }
            gaemi_annotations.append(record)

        # save JSON
        record_path = os.path.join(record_path, f'semantic_{target_type}_annotations.json')
        return record_path, gaemi_annotations

    def create_semantic_png(self, annotation_info, target_type='train'):
        self.logger.info(f"Creating semantic PNG files for {target_type} set...")
        self.logger.info(f"PNG files will use contiguous_id (1 to {len(self.dataset_id_to_contiguous_id)})")
        self.logger.info(f"Non-trainable classes will be stored as 255 (ignore label)")

        skipped_labels = set()
        non_trainable_count = 0

        for annotation in tqdm(annotation_info):
            json_path = annotation['json_file_name']
            sem_seg_path = annotation['sem_seg_file_name']

            # Read JSON file
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # Get image size
            height = annotation['height']
            width = annotation['width']

            # Create Semantic segmentation PNG (initialize 255 - ignore label)
            semantic_map = np.full((height, width), 255, dtype=np.uint8)

            # Process each shape (polygon)
            for shape in json_data.get('shapes', []):
                label = shape['label']
                points = shape['points']

                # Apply class_remap when data type is DrivingAreaSegmentation
                if self.class_remap.get(self.data_type, None):
                    if label in self.class_remap[self.data_type]:
                        original_label = label
                        label = self.class_remap[self.data_type][label]
                        self.logger.debug(f"Remapped {original_label} -> {label}")

                # Find ID from class information
                if label not in self.class_info:
                    if label not in skipped_labels:
                        self.logger.warning(f"Unknown label '{label}' in {json_path}, skipping...")
                        skipped_labels.add(label)
                    continue

                dataset_id = self.class_info[label]['id']
                is_trainable = self.class_info[label].get('trainable', True)

                # Store non-trainable classes as 255 (ignore label)
                if not is_trainable:
                    class_id = 255
                    non_trainable_count += 1
                    self.logger.debug(f"Non-trainable class '{label}' (dataset_id={dataset_id}) -> 255 (ignore)")
                # Convert trainable classes to contiguous_id
                elif dataset_id in self.dataset_id_to_contiguous_id:
                    class_id = self.dataset_id_to_contiguous_id[dataset_id]
                    self.logger.debug(f"Trainable class '{label}' (dataset_id={dataset_id}) -> contiguous_id={class_id}")
                else:
                    # For unmapped cases (safety fallback)
                    class_id = 255
                    self.logger.warning(f"Class '{label}' (dataset_id={dataset_id}) not in mapping, using 255 (ignore)")

                # Convert polygon coordinates to numpy array
                pts = np.array(points, dtype=np.int32)

                # Draw polygon on semantic map
                cv2.fillPoly(semantic_map, [pts], class_id)

            # Create save directory
            os.makedirs(os.path.dirname(sem_seg_path), exist_ok=True)

            # Save as PNG file
            cv2.imwrite(sem_seg_path, semantic_map)
            self.logger.debug(f"Saved semantic PNG: {sem_seg_path}")

        if skipped_labels:
            self.logger.warning(f"Skipped unknown labels: {skipped_labels}")
        if non_trainable_count > 0:
            self.logger.info(f"Converted {non_trainable_count} non-trainable polygons to ignore label (255)")

        self.logger.info(f"Semantic PNG creation completed for {target_type} set!")
        self.logger.info(f"  - PNG values: contiguous_id [1-{len(self.dataset_id_to_contiguous_id)}] and 255 (ignore)")
        self.logger.info(f"  - Model expects: NUM_CLASSES = {len(self.dataset_id_to_contiguous_id)}")


