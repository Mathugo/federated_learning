# tensorflow 2 model :
pip install openvino_dev==2021.4.2
# or 
pip install openvino_dev[tensorflow]
# convert to IR
# get user site ---> PYTHON_SITE_PACKAGES>/openvino/tools/mo/front/tf/ssd_support_api_v.2.4.json
python -m site --user-site
# --> /home/cloud/.local/lib/python3.9/site-packages/

# anaconda3 -> pip list | tail -n +3 | xargs -exec pip show
# IR Representation
mo --saved_model_dir my_ssd_mobilenet320_v2_fpn_10000_mAP_ng_0_98_g_0_0085/saved_model --tensorflow_object_detection_api_pipeline_config my_ssd_mobilenet320_v2_fpn_10000_mAP_ng_0_98_g_0_0085/pipeline.config --input_shape "[1, 320, 320, 3]" --output_dir .