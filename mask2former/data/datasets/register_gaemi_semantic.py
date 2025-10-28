import os
import re
import json
import logging
import yaml

from detectron2.data import DatasetCatalog, MetadataCatalog
from custom_util.config.class_config import get_class_info

logger = logging.getLogger(__name__)

def load_custom_dicts(service_areas, target_img_json_path, gt_json_path):
    """
    Load dataset from semantic annotation JSON, filtered by record JSON.
    Uses sem_seg_file_name to reference existing PNG GT files.

    Args:
        dataset_dir: Not used (kept for compatibility)
        class_names: List of class names (for validation)
        service_areas: List of service areas to use (empty = use all)
        target_img_json_path: Path to record JSON (e.g., record_train.json)
        dataset_id_to_contiguous_id: Mapping from original dataset ID to contiguous ID
        ignore_label: Label value for non-trainable classes (default: 255)

    Returns:
        List[dict]: Detectron2 standard dataset format
    """
    logger.info(f"Loading semantic dataset from {target_img_json_path}...")

    # Step 1: Read record JSON to get image paths
    with open(target_img_json_path, 'r') as f:
        record_data = json.load(f)

    # Step 2: Filter by service areas
    if not service_areas or len(service_areas) == 0:
        # Empty list = use all service areas
        selected_areas = list(record_data.keys())
        logger.info(f"Using all service areas: {selected_areas}")
    else:
        # Use only specified service areas
        selected_areas = [area for area in service_areas if area in record_data]
        logger.info(f"Using selected service areas: {selected_areas}")

    # Step 3: Collect all image paths from selected areas
    target_image_paths = set()
    for area in selected_areas:
        paths = record_data.get(area, [])
        target_image_paths.update(paths)
        logger.info(f"Service area '{area}': {len(paths)} images")

    logger.info(f"Total images to load: {len(target_image_paths)}")

    if not os.path.exists(gt_json_path):
        raise FileNotFoundError(
            f"Semantic annotation file not found: {gt_json_path}\n"
            f"Please make sure the file exists in the same directory as {target_img_json_path}"
        )

    # Step 4: Load semantic annotations JSON
    logger.info(f"Loading semantic annotations from {gt_json_path}...")
    with open(gt_json_path, 'r') as f:
        semantic_data = json.load(f)

    logger.info(f"Total annotations in semantic JSON: {len(semantic_data)}")

    # Step 5: Filter and convert annotations
    dataset_dicts = []
    loaded_count = 0
    skipped_count = 0
    missing_field_count = 0
    missing_sem_seg_count = 0

    for item in semantic_data:
        file_name = item.get('file_name', '')
        
        if not file_name:
            logger.warning(f"Item missing 'file_name' field, skipping")
            missing_field_count += 1
            continue

        # Only include images that are in the record JSON
        if file_name not in target_image_paths:
            skipped_count += 1
            continue

        # Validate required fields
        if 'sem_seg_file_name' not in item:
            logger.warning(f"No sem_seg_file_name found for {file_name}, skipping")
            missing_field_count += 1
            continue
        
        # Validate sem_seg file actually exists
        sem_seg_gt_path = item['sem_seg_file_name']
        if not os.path.exists(sem_seg_gt_path):
            logger.warning(f"Semantic segmentation file not found: {sem_seg_gt_path} (image: {file_name})")
            missing_sem_seg_count += 1
            continue

        # Validate other required fields
        if 'height' not in item or 'width' not in item:
            logger.warning(f"Missing height/width for {file_name}, skipping")
            missing_field_count += 1
            continue

        # Create Detectron2 format record
        record = {
            "file_name": item['file_name'],
            "image_id": item.get('image_id', loaded_count),
            "height": item['height'],
            "width": item['width'],
            "sem_seg_file_name": item['sem_seg_file_name'],  # PNG GT 경로
        }

        dataset_dicts.append(record)
        loaded_count += 1

    logger.info(f"Successfully loaded {loaded_count} images with semantic annotations")
    logger.info(f"Skipped {skipped_count} images (not in record JSON)")
    if missing_field_count > 0:
        logger.warning(f"Skipped {missing_field_count} images (missing required fields)")
    if missing_sem_seg_count > 0:
        logger.warning(f"Skipped {missing_sem_seg_count} images (sem_seg file not found)")

    return dataset_dicts

def register_gaemi_dataset(cfg, name, target_json_path):
    # 1. get class names and dataset directory from config
    # Define class names in YAML config like 'DATASETS.CLASS_NAMES: ["class1", "class2", ...]'
    try:
        class_names = cfg.DATASETS.CLASS_NAMES
    except AttributeError:
        raise AttributeError("CLASS_NAMES not found in config! Please add `DATASETS.CLASS_NAMES` to your YAML config file.")

    dataset_dir = cfg.DATASETS.DATA_DIR_PATH
    if not dataset_dir:
        raise ValueError("DATA_DIR_PATH not found in config! Please add `DATASETS.DATA_DIR_PATH` to your YAML config file.")

    available_service_areas = cfg.DATASETS.TARGET_SERVICE_AREAS

    # 2. Filter trainable classes from class_info
    # Only classes with 'trainable': True will be included in training
    class_info = get_class_info()
    trainable_classes = [class_name for class_name in class_names 
                         if class_info.get(class_name, {}).get('trainable', True)]
    
    non_trainable_classes = [class_name for class_name in class_names 
                            if not class_info.get(class_name, {}).get('trainable', True)]
    
    logger.info(f"Trainable classes ({len(trainable_classes)}): {trainable_classes}")
    logger.info(f"Non-trainable classes ({len(non_trainable_classes)}): {non_trainable_classes}")

    # 3. Create ID mappings for trainable classes only
    # Original dataset_id -> contiguous_id (1, 2, 3, ...)
    # Note: contiguous_id starts from 1 because 0 is reserved for background/ignore in the model
    dataset_id_to_contiguous_id = {}
    contiguous_id = 1  # Start from 1 instead of 0
    # for trainable classes only
    for class_name in trainable_classes:
        if class_name in class_info:
            original_id = class_info[class_name]['id']
            dataset_id_to_contiguous_id[original_id] = contiguous_id
            contiguous_id += 1

    # 4. register dataset
    base_dir = os.path.dirname(target_json_path)
    base_name = os.path.basename(target_json_path)

    if "train" in base_name:
        semantic_json_name = "semantic_train_annotations.json"
    elif "val" in base_name:
        semantic_json_name = "semantic_val_annotations.json"
    elif "test" in base_name:
        semantic_json_name = "semantic_test_annotations.json"
    gt_json_path = os.path.join(base_dir, semantic_json_name)

    ignore_label = 255
    DatasetCatalog.register(
        name,
        lambda: load_custom_dicts(
            available_service_areas,
            target_json_path,
            gt_json_path,
        )
    )

    # 5. register metadata - Semantic Segmentation
    MetadataCatalog.get(name).set(
        # necessary: class information (only trainable classes)
        all_classes=class_names,               # all classes
        stuff_classes=trainable_classes,       # semantic classes
        thing_classes=trainable_classes,       # for compatibility

        # necessary: evaluation settings
        evaluator_type="gaemi_semantic",       # custom evaluator type
        ignore_label=ignore_label,

        # necessary: ID mappings (for evaluator)
        # not necessary to separate stuff/thing mappings in semantic segmentation
        all_dataset_id_to_contiguous_id=dataset_id_to_contiguous_id,
        stuff_dataset_id_to_contiguous_id=dataset_id_to_contiguous_id,
        thing_dataset_id_to_contiguous_id=dataset_id_to_contiguous_id,

        # optional: path information
        image_root=dataset_dir,
        gt_json_path=gt_json_path,  # semantic_gt_json_path
        val_img_json_path=cfg.DATASETS.TEST_JSON_PATH,
        class_info=class_info,
        available_service_areas=available_service_areas,
    )
    logger.info(f"Registered dataset '{name}' with {len(trainable_classes)} trainable classes (total: {len(class_names)}).")

    # Debug: Print metadata to verify registration
    logger.info(f"Metadata for '{name}':")
    meta = MetadataCatalog.get(name)
    logger.info(f"  - stuff_classes ({len(meta.stuff_classes)}): {meta.stuff_classes}")
    logger.info(f"  - thing_classes ({len(meta.thing_classes)}): {meta.thing_classes}")
    logger.info(f"  - ignore_label: {meta.ignore_label}")
    logger.info(f"  - evaluator_type: {meta.evaluator_type}")
    logger.info(f"  - ID mapping: {dataset_id_to_contiguous_id}")


def register_all_gaemi(config_path=None):
    """
    Register GAEMI semantic datasets automatically for inference without cfg.
    Reads configuration from Base-Gaemi-SemanticSegmentation.yaml

    Args:
        config_path: Path to config file. If None, uses default path.
    """
    # Determine config path
    if config_path is None:
        # Default path relative to Mask2Former directory
        # __file__ is: .../Mask2Former/mask2former/data/datasets/register_gaemi_semantic.py
        # Need to go up 4 levels: datasets -> data -> mask2former -> Mask2Former
        current_file = os.path.abspath(__file__)
        mask2former_module_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # Mask2Former/mask2former
        mask2former_root = os.path.dirname(mask2former_module_dir)  # Mask2Former

        ### 확인 필수 ####
        config_path = os.path.join(
            mask2former_root,
            "configs/gaemi/semantic-segmentation/Base-Gaemi-SemanticSegmentation.yaml"
        )
        #############3

    if not os.path.exists(config_path):
        logger.warning(f"GAEMI semantic config file not found: {config_path}")
        logger.warning("Skipping automatic GAEMI semantic dataset registration.")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        # Read YAML config and extract only needed fields using regex
        # to avoid issues with python objects in other sections
        with open(config_path, 'r') as f:
            content = f.read()

        # Extract DATA_DIR_PATH using regex
        data_dir_match = re.search(r'DATA_DIR_PATH:\s*["\']?([^"\'\n]+)["\']?', content)
        data_dir_path = data_dir_match.group(1).strip() if data_dir_match else ''

        # Extract CLASS_NAMES - look for the list section
        class_names = []
        class_names_match = re.search(r'CLASS_NAMES:\s*\[(.*?)\]', content, re.DOTALL)
        if class_names_match:
            # Inline list format
            list_content = class_names_match.group(1)
            class_names = [c.strip().strip('"\',') for c in list_content.split(',') if c.strip() and not c.strip().startswith('#')]
        else:
            # Multi-line list format
            class_names_section = re.search(r'CLASS_NAMES:\s*\n((?:\s*-\s*[^\n]+\n)+)', content)
            if class_names_section:
                for line in class_names_section.group(1).split('\n'):
                    match = re.search(r'-\s*["\']?([^"\',\n]+)["\']?', line)
                    if match:
                        class_name = match.group(1).strip()
                        if class_name and not class_name.startswith('#'):
                            class_names.append(class_name)

        logger.info(f"Parsed config - Data directory: {data_dir_path}, Classes found: {len(class_names)}")

        if not class_names:
            logger.warning("CLASS_NAMES not found in config file. Skipping GAEMI semantic registration.")
            return

        logger.info(f"Auto-registering GAEMI semantic datasets from config: {config_path}")
        logger.info(f"  - Data directory: {data_dir_path}")
        logger.info(f"  - Classes: {len(class_names)}")

        # Filter trainable classes from class_info
        class_info = get_class_info()
        trainable_classes = [class_name for class_name in class_names 
                            if class_info.get(class_name, {}).get('trainable', True)]

        # Create ID mappings for trainable classes only
        # Note: contiguous_id starts from 1 because 0 is reserved for background/ignore in the model
        dataset_id_to_contiguous_id = {}
        contiguous_id = 1  # Start from 1 instead of 0

        for class_name in trainable_classes:
            if class_name in class_info:
                original_id = class_info[class_name]['id']
                dataset_id_to_contiguous_id[original_id] = contiguous_id
                contiguous_id += 1

        # Register basic metadata for common dataset names
        for split in ["train", "val", "test"]:
            dataset_name = f"gaemi_{split}"

            # Only register metadata (not DatasetCatalog for inference)
            MetadataCatalog.get(dataset_name).set(
                stuff_classes=trainable_classes,
                thing_classes=trainable_classes,
                stuff_dataset_id_to_contiguous_id=dataset_id_to_contiguous_id,  # Semantic segmentation용
                thing_dataset_id_to_contiguous_id=dataset_id_to_contiguous_id,  # Mask2Former 모델 호환용
                evaluator_type="gaemi_semantic",
                ignore_label=255,
                image_root=data_dir_path,
            )

            logger.info(f"Registered metadata for '{dataset_name}' with {len(trainable_classes)} trainable classes")

    except Exception as e:
        logger.warning(f"Failed to auto-register GAEMI semantic datasets: {e}")
        logger.warning("You may need to register manually in your training script.")


# Auto-register GAEMI semantic datasets when this module is imported
if __name__.endswith(".register_gaemi_semantic"):
    register_all_gaemi()

