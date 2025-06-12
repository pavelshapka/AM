run_train:
	python3 main.py --config configs/am/cifar/generation.py --workdir $PWD/checkpoint/${SLURM_JOB_ID} --mode 'train'

run_eval:
	python3 main.py --config configs/am/cifar/generation.py --workdir $PWD/checkpoint/${SLURM_JOB_ID} --mode 'eval'

run_fid-stats:
	python3 main.py --config configs/am/cifar/generation.py --workdir $PWD/checkpoint/${SLURM_JOB_ID} --mode 'fid_stats'