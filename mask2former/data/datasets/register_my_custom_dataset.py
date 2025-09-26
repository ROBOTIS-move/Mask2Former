
import os
import json
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog

logger = logging.getLogger(__name__)

def load_custom_dicts(dataset_dir, class_names):
    """
    dataset_dir: JSON 파일들이 있는 디렉토리 경로
    class_names: 클래스 이름 리스트

    Detectron2의 표준 데이터셋 형식(list[dict])을 반환합니다.
    """
    logger.info(f"Loading custom dataset from {dataset_dir}...")
    dataset_dicts = []
    
    json_files = [f for f in os.listdir(dataset_dir) if f.endswith('.json')]
    
    for idx, json_file in enumerate(json_files):
        json_path = os.path.join(dataset_dir, json_file)
        
        with open(json_path) as f:
            img_anns = json.load(f)

        record = {}
        
        # 이미지 파일 경로는 JSON 파일과 동일한 디렉토리에 있다고 가정합니다.
        # 만약 다른 곳에 있다면 이 부분을 수정해야 합니다.
        image_path = os.path.join(dataset_dir, img_anns["imagePath"])
        
        record["file_name"] = image_path
        record["image_id"] = idx
        record["height"] = img_anns["imageHeight"]
        record["width"] = img_anns["imageWidth"]
      
        annos = []
        for shape in img_anns["shapes"]:
            # labelme 형식의 polygon 포인트를 Detectron2 형식으로 변환
            px = [p[0] for p in shape["points"]]
            py = [p[1] for p in shape["points"]]
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            # 클래스 이름을 ID로 변환
            try:
                category_id = class_names.index(shape["label"])
            except ValueError:
                # class_names에 없는 레이블은 건너뜁니다.
                logger.warning(f"Label '{shape['label']}' in {json_file} is not in the configured CLASS_NAMES. Skipping.")
                continue

            obj = {
                "segmentation": [poly],
                "category_id": category_id,
            }
            annos.append(obj)
        
        record["annotations"] = annos
        dataset_dicts.append(record)
        
    logger.info(f"Loaded {len(dataset_dicts)} images.")
    return dataset_dicts

def register_my_custom_dataset(name, cfg, dataset_dir):
    """
    데이터셋을 DatasetCatalog와 MetadataCatalog에 등록합니다.
    클래스 정보는 cfg 객체에서 가져옵니다.
    """
    
    # 1. 설정 파일에서 클래스 이름 가져오기
    # YAML 파일에 `DATASETS.CLASS_NAMES: ["class1", "class2", ...]` 와 같이 정의해야 합니다.
    try:
        class_names = cfg.DATASETS.CLASS_NAMES
    except AttributeError:
        raise AttributeError("CLASS_NAMES not found in config! Please add `DATASETS.CLASS_NAMES` to your YAML config file.")

    # 2. 데이터셋 등록
    DatasetCatalog.register(name, lambda: load_custom_dicts(dataset_dir, class_names))
    
    # 3. 메타데이터 등록
    MetadataCatalog.get(name).set(
        stuff_classes=class_names,
        thing_classes=class_names,
        evaluator_type="sem_seg", 
        ignore_label=255,
    )
    logger.info(f"Registered dataset '{name}' with {len(class_names)} classes.")

# 데이터셋 이름과 경로 설정
# 예시: register_my_custom_dataset("my_dataset_train", "/path/to/your/train/data")
# 예시: register_my_custom_dataset("my_dataset_val", "/path/to/your/val/data")

# 이 파일이 임포트될 때 데이터셋을 자동으로 등록하려면 아래와 같이 호출할 수 있습니다.
# 실제 경로로 수정해야 합니다.
# from detectron2.utils.comm import is_main_process
# if is_main_process():
#     register_my_custom_dataset("my_custom_train", "path/to/your/training/jsons")
#     register_my_custom_dataset("my_custom_val", "path/to/your/validation/jsons")

