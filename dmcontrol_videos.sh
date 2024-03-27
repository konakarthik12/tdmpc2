CUDA_VISIBLE_DEVICES=0 python tdmpc2/train.py task=dog-run steps=1000 disable_wandb=False wandb_project=sanghyun_son wandb_entity=shh1295 eval_episodes=1
CUDA_VISIBLE_DEVICES=0 python tdmpc2/train.py task=manipulator-bring_ball steps=1000 disable_wandb=False wandb_project=sanghyun_son wandb_entity=shh1295 eval_episodes=1
CUDA_VISIBLE_DEVICES=0 python tdmpc2/train.py task=humanoid-stand steps=1000 disable_wandb=False wandb_project=sanghyun_son wandb_entity=shh1295 eval_episodes=1
CUDA_VISIBLE_DEVICES=0 python tdmpc2/train.py task=finger-spin steps=1000 disable_wandb=False wandb_project=sanghyun_son wandb_entity=shh1295 eval_episodes=1
CUDA_VISIBLE_DEVICES=0 python tdmpc2/train.py task=fish-swim steps=1000 disable_wandb=False wandb_project=sanghyun_son wandb_entity=shh1295 eval_episodes=1
