import os
import logging
import numpy as np
import cv2

from PIL import Image
from tqdm import tqdm


class PanopticUtil:
    
    def __init__(self, config, class_info, class_remap):
        self.gaemi_config = config
        self.class_info = class_info
        self.class_remap = class_remap

        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s][%(levelname)s][%(name)s]: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info('PanopticUtil initialized')

    def make_panoptic_annotations(self, data_list, target_type='train'):
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

        record_path = self.gaemi_config.get('record_path', '')
        save_path = os.path.join(record_path, f'panoptic_{target_type}_annotations.json')
        return save_path, gaemi_annotations

    def _calculate_annotation(self, annotation_np, json_path):
        json_info = self._read_json(json_path)
        image_height = json_info.get('imageHeight', annotation_np.shape[0])
        image_width = json_info.get('imageWidth', annotation_np.shape[1])

        annotations = []

        for shp in json_info['shapes']:
            label = shp['label']

            # Convert label if in remap (e.g., 'cement' -> 'road')
            converted_label = self.class_remap.get(label, label)

            # Get label info from class_info
            try:
                label_info = self.class_info[converted_label]
            except KeyError:
                self.logger.warning(f'Label "{label}" (converted: "{converted_label}") not found in class_info. Skipping.')
                continue

            label_id = label_info.get('id', 0)
            if label_id == 0:
                self.logger.warning(f'Label "{label}" has id=0. Skipping.')
                continue

            # Get polygon points and flatten to single list
            polygon = shp['points']  # [[x1, y1], [x2, y2], [x3, y3]]
            flatten_polygon = [coord for point in polygon for coord in point]  # [x1, y1, x2, y2, x3, y3]
            # Calculate area and bbox
            area, bbox = self._calculate_area_and_bbox(polygon, image_height, image_width)

            # iscrowd is using only instance segmentation, default is 0 for panoptic
            iscrowd = 0

            # Stuff 클래스: instance_id = 0 (같은 클래스는 하나의 영역)
            # Panoptic ID = category_id * 1000 + instance_id
            instance_id = 0
            panoptic_id = label_id * 1000 + instance_id

            annotation_info = {
                'label': label,  # 디버깅용
                "id": panoptic_id,  # Panoptic ID (category_id * 1000 + instance_id)
                "category_id": int(label_id),  # Dataset ID (1~25)
                'segmentation': [flatten_polygon],  # Detectron2 format: [[x1,y1,...]]
                "area": int(area),
                "bbox": bbox,
                "iscrowd": iscrowd
            }

            annotations.append(annotation_info)

        return annotations
    
    def _calculate_area_and_bbox(self, polygon, image_height, image_width):
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
    
    def create_panoptic_png(self, target_type='train'):
        """
        각 이미지에 대해 panoptic PNG 파일 생성
        RGB 인코딩: R + G*256 + B*256^2 = panoptic_id
        panoptic_id = category_id * 1000 + instance_id
        
        Stuff 클래스: 같은 클래스는 instance_id=0 (하나의 영역)
        Thing 클래스: 같은 클래스라도 instance_id++ (개별 인스턴스 구분)
        """
        self.logger.info(f"Creating panoptic PNG files for {target_type} set...")
        
        target_paths = self._get_target_file_path(target_type)
        
        # panoptic PNG 저장 디렉토리
        panoptic_dir = os.path.join(
            self.gaemi_config.get('record_path', ''),
            f'panoptic_{target_type}'
        )
        os.makedirs(panoptic_dir, exist_ok=True)
        
        for img_path in tqdm(target_paths, desc=f'Creating panoptic PNGs ({target_type})'):
            # annotation JSON 로드
            json_path = img_path.replace('images', 'labels').replace('.jpg', '.json')
            if not os.path.exists(json_path):
                self.logger.warning(f"JSON not found: {json_path}")
                continue
                
            json_data = self._read_json(json_path)
            height = json_data.get('imageHeight', 0)
            width = json_data.get('imageWidth', 0)
            
            if height == 0 or width == 0:
                self.logger.warning(f"Invalid image size in {json_path}")
                continue
            
            # Panoptic ID 맵 생성
            panoptic_map = np.zeros((height, width), dtype=np.int32)
            
            # 클래스별 instance_id 카운터 (thing 클래스용)
            # Gaemi는 현재 모든 클래스를 stuff로 처리하므로 instance_id=0 사용
            class_instance_counter = {}
            
            for shape in json_data.get('shapes', []):
                label = shape.get('label', '')
                
                # Convert label if in remap
                converted_label = self.class_remap.get(label, label)
                
                # Get label info from class_info
                if converted_label not in self.class_info:
                    self.logger.warning(f'Label "{label}" (converted: "{converted_label}") not found in class_info. Skipping.')
                    continue
                
                label_info = self.class_info[converted_label]
                label_id = label_info.get('id', 0)
                
                if label_id == 0:
                    self.logger.warning(f'Label "{label}" has id=0 (void). Skipping.')
                    continue
                
                # Stuff 클래스: instance_id = 0 (같은 클래스는 하나의 영역)
                # Thing 클래스: instance_id++ (개별 인스턴스 구분)
                # 현재 Gaemi는 모든 클래스를 stuff로 처리
                instance_id = 0
                
                # Panoptic ID = category_id * 1000 + instance_id
                panoptic_id = label_id * 1000 + instance_id
                
                # Polygon을 mask로 변환
                points = shape.get('points', [])
                if len(points) < 3:
                    continue
                    
                polygon = np.array(points, dtype=np.int32)
                cv2.fillPoly(panoptic_map, [polygon], panoptic_id)
            
            # RGB 인코딩하여 PNG 저장
            rgb_panoptic = self._id_to_rgb(panoptic_map)
            
            base_name = os.path.basename(img_path).replace('.jpg', '.png')
            output_path = os.path.join(panoptic_dir, base_name)
            Image.fromarray(rgb_panoptic).save(output_path)
        
        self.logger.info(f"Created {len(target_paths)} panoptic PNG files in {panoptic_dir}")

    def _id_to_rgb(self, panoptic_id):
        """
        Convert panoptic ID to RGB encoding (without panopticapi dependency)
        
        Args:
            panoptic_id: int32 array of panoptic IDs
            
        Returns:
            RGB uint8 array [H, W, 3]
        """
        # RGB encoding: R + G*256 + B*256^2 = panoptic_id
        rgb = np.zeros((*panoptic_id.shape, 3), dtype=np.uint8)
        rgb[:, :, 0] = panoptic_id % 256
        rgb[:, :, 1] = (panoptic_id // 256) % 256
        rgb[:, :, 2] = (panoptic_id // (256 ** 2)) % 256
        return rgb