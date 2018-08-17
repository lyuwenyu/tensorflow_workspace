rm -rf ./inference/*

CUDA_VISIBLE_DEVICES="" python export_inference_graph.py --input_type=image_tensor --pipeline_config_path=./config/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix=./outputs/ssd_mobil_v2/train/model.ckpt-11084 --output_directory=inference/

CUDA_VISIBLE_DEVICES="" python test.py
