# test lasher
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=NVL python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mfja --dataset_name LasHeR --yaml_name rgbt-4e-4

# test rgbt234
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_LEVEL=NVL python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mfja --dataset_name RGBT234 --yaml_name rgbt-4e-4