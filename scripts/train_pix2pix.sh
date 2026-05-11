set -ex
python train.py --dataroot ./datasets/facades --name facades_pix2pix_frobenius --model pix2pix --netG unet_256 --direction BtoA --lambda_L1 100 --lambda_frobenius 0.1 --dataset_mode aligned --norm batch --pool_size 0  --use_wandb
