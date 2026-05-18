export PYTHONPATH="$PWD:$PYTHONPATH"
export TMPDIR="/user/ganesang/cvl/DACv2/cache"
export WANDB_HOME=${TMPDIR}
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

python scripts/train.py --config-file configs/train/hm3d+taskonomy+hypersim+ddad+lyft+argoverse2+a2d2_unidac_dinov3l.json --base-path datasets --distributed --model-name UniDAC
