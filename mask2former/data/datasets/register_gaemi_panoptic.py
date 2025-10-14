
import os
import json
import logging

from detectron2.data import DatasetCatalog, MetadataCatalog
from custom_util.config.class_config import class_info

logger = logging.getLogger(__name__)

def load_custom_dicts(dataset_dir, class_names, service_areas, target_img_json_path, dataset_id_to_contiguous_id, ignore_label=255):
    """
    Load dataset from panoptic annotation JSON, filtered by record JSON.
    Wraps segmentation in outer list for Detectron2 format.

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
    logger.info(f"Loading custom dataset from {target_img_json_path}...")

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

    # Step 4: Determine panoptic annotation JSON path
    # record_train.json -> panoptic_train_annotations.json
    # record_val.json -> panoptic_val_annotations.json
    base_dir = os.path.dirname(target_img_json_path)
    base_name = os.path.basename(target_img_json_path)

    if "train" in base_name:
        panoptic_json_name = "panoptic_train_annotations.json"
    elif "val" in base_name:
        panoptic_json_name = "panoptic_val_annotations.json"
    elif "test" in base_name:
        panoptic_json_name = "panoptic_test_annotations.json"
    else:
        panoptic_json_name = "panoptic_annotations.json"

    panoptic_json_path = os.path.join(base_dir, panoptic_json_name)

    if not os.path.exists(panoptic_json_path):
        raise FileNotFoundError(
            f"Panoptic annotation file not found: {panoptic_json_path}\n"
            f"Please make sure the file exists in the same directory as {target_img_json_path}"
        )

    # Step 5: Load panoptic annotations JSON
    logger.info(f"Loading panoptic annotations from {panoptic_json_path}...")
    with open(panoptic_json_path, 'r') as f:
        panoptic_data = json.load(f)

    logger.info(f"Total annotations in panoptic JSON: {len(panoptic_data)}")

    # Step 6: Filter and convert annotations
    dataset_dicts = []
    loaded_count = 0
    skipped_count = 0

    for item in panoptic_data:
        file_name = item['file_name']

        # Only include images that are in the record JSON
        if file_name not in target_image_paths:
            skipped_count += 1
            continue

        # Validate required fields
        if 'annotations' not in item:
            logger.warning(f"No annotations found for {file_name}, skipping")
            continue

        # Create Detectron2 format record
        record = {
            "file_name": item['file_name'],
            "image_id": item.get('image_id', loaded_count),
            "height": item['height'],
            "width": item['width'],
            "annotations": []
        }

        # Process annotations
        for ann in item['annotations']:
            # Validate annotation has required fields
            required_fields = ['segmentation', 'bbox', 'category_id', 'area', 'iscrowd']
            if not all(field in ann for field in required_fields):
                logger.warning(f"Annotation missing required fields in {file_name}, skipping")
                continue

            # Convert segmentation format
            # From: [x1, y1, x2, y2, x3, y3, ...]
            # To: [[x1, y1, x2, y2, x3, y3, ...]]
            segmentation = ann['segmentation']

            if not isinstance(segmentation, list) or len(segmentation) < 6:  # At least 3 points (6 values)
                logger.warning(f"Invalid segmentation in {file_name}: requires at least 3 points, skipping")
                continue

            # Simply wrap in outer list for Detectron2 format
            detectron2_segmentation = [segmentation]

            # Convert dataset category_id to contiguous_id
            # Non-trainable classes (void, forbidden-area) are mapped to ignore_label
            original_category_id = ann['category_id']
            contiguous_id = dataset_id_to_contiguous_id.get(original_category_id, ignore_label)

            # Skip non-trainable classes (void, forbidden-area)
            # These annotations should not be included in the dataset
            if contiguous_id == ignore_label:
                continue

            obj = {
                "segmentation": detectron2_segmentation,  # [[x1,y1,x2,y2,...]]
                "bbox": ann['bbox'],                      # [x, y, w, h]
                "bbox_mode": ann.get('bbox_mode', 1),     # 1=XYWH_ABS (default for panoptic)
                "area": ann['area'],
                "category_id": contiguous_id,             # Use contiguous_id for trainable classes
                "iscrowd": ann['iscrowd']
            }
            record["annotations"].append(obj)

        # Only add records with at least one annotation
        if len(record["annotations"]) > 0:
            dataset_dicts.append(record)
            loaded_count += 1

    logger.info(f"Successfully loaded {loaded_count} images with annotations")
    logger.info(f"Skipped {skipped_count} images (not in record JSON)")

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
    trainable_classes = [class_name for class_name in class_names 
                         if class_info.get(class_name, {}).get('trainable', True)]
    
    non_trainable_classes = [class_name for class_name in class_names 
                            if not class_info.get(class_name, {}).get('trainable', True)]
    
    logger.info(f"Trainable classes ({len(trainable_classes)}): {trainable_classes}")
    logger.info(f"Non-trainable classes ({len(non_trainable_classes)}): {non_trainable_classes}")

    # 3. Create ID mappings for trainable classes only
    # Original dataset_id -> contiguous_id (0, 1, 2, ...)
    dataset_id_to_contiguous_id = {}
    contiguous_id = 0
    
    for class_name in trainable_classes:
        if class_name in class_info:
            original_id = class_info[class_name]['id']
            dataset_id_to_contiguous_id[original_id] = contiguous_id
            contiguous_id += 1
    
    # Reverse mapping for metadata
    thing_dataset_id_to_contiguous_id = {k: v for k, v in dataset_id_to_contiguous_id.items()}
    stuff_dataset_id_to_contiguous_id = {k: v for k, v in dataset_id_to_contiguous_id.items()}

    # 4. register dataset
    ignore_label = 255
    DatasetCatalog.register(
        name[0],
        lambda: load_custom_dicts(
            dataset_dir,
            trainable_classes,  # Pass only trainable classes
            available_service_areas,
            target_json_path,
            dataset_id_to_contiguous_id,  # Pass ID mapping
            ignore_label
        )
    )

    # 5. register metadata - Panoptic Segmentation
    MetadataCatalog.get(name[0]).set(
        # necessary: class information (only trainable classes)
        thing_classes=trainable_classes,                     # individual object classes
        stuff_classes=trainable_classes,                     # background/area classes

        # necessary: evaluation settings
        evaluator_type="gaemi",                              # custom evaluator type
        ignore_label=ignore_label,

        # necessary: ID mappings
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,

        # optional: path information
        image_root=dataset_dir,
        val_img_json_path=cfg.DATASETS.TEST_JSON_PATH,
        available_service_areas=available_service_areas,

        # Panoptic only (required for panoptic segmentation)
        label_divisor=1000,                                  # classify panoptic IDs
    )
    logger.info(f"Registered dataset '{name[0]}' with {len(trainable_classes)} trainable classes (total: {len(class_names)}).")

    # Debug: Print metadata to verify registration
    logger.info(f"Metadata for '{name[0]}':")
    meta = MetadataCatalog.get(name[0])
    logger.info(f"  - thing_classes ({len(meta.thing_classes)}): {meta.thing_classes}")
    logger.info(f"  - stuff_classes ({len(meta.stuff_classes)}): {meta.stuff_classes}")
    logger.info(f"  - ignore_label: {meta.ignore_label}")
    logger.info(f"  - evaluator_type: {meta.evaluator_type}")
    logger.info(f"  - ID mapping: {dataset_id_to_contiguous_id}")
