# Training Evi
set -e
lr=4e-4  #学习率
NCCL_P2P_LEVEL=NVL python tracking/train.py --script mfja --config rgbt-$lr --save_dir ./output --mode multiple --nproc_per_node 3
mkdir ./models
ckpt_path="./output/checkpoints/train/mfja/rgbt-$lr/ETrack_ep0020.pth.tar"
cp $ckpt_path ./models
mv ./models/ETrack_ep0020.pth.tar ./models/ETrack_rgbt-4e-4.pth.tar  # 重命名
CUDA_VISIBLE_DEVICES=0,1,2 NCCL_P2P_LEVEL=NVL python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mfja --dataset_name RGBT234 --yaml_name rgbt-$lr --threads 3
CUDA_VISIBLE_DEVICES=0,1,2 NCCL_P2P_LEVEL=NVL python ./RGBT_workspace/test_rgbt_mgpus.py --script_name mfja --dataset_name LasHeR --yaml_name rgbt-$lr --threads 3
ym=$(date +"%m-%d")
backup_folder=$(date +"%Y-%m-%d_%H:%M:%S")
zip_name="result-"$ym"-3layers-fusion-iner-adap-ep20-$lr.zip"  #! change zip file name before run
zip -r $zip_name ./RGBT_workspace/results
zip -r $zip_name ./tensorboard
zip -r $zip_name $ckpt_path
if [ -d "output" ]; then
    backup_folder=$(date +"%Y-%m-%d_%H:%M:%S")
    backup_folder=./backup/$backup_folder
    mkdir $backup_folder
    mv ./output $backup_folder
    mv ./tensorboard $backup_folder
    mv ./RGBT_workspace/results $backup_folder
    mv ./models $backup_folder
fi
if [ -f "$zip_name" ]; then  # 如果压缩完毕
    oss cp $zip_name oss://
    rm $zip_name
    shutdown
fi