# Collecting Trajectory

If not cloned this repository recursively, initialize submodule for SB3 support.

```bash
git submodule init
git submodule update
```

Then, go to `external/traj_collect`, and run following command to install SB3.

```bash
pip install -e .
```

Finally, run following commands to test trajectory collection. See `tdmpc2/collect.py` for more details.

```bash
python tdmpc2/collect.py task=dog_run sb3_algo=ppo steps=1000       # regular tdmpc2 env
python tdmpc2/collect.py task=t2a_ant sb3_algo=sac steps=1000       # t2a env, which also stores graph node & edge info
```

Find the saved trajectory and the parsed version of it at the `outputs` directory.