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
python tdmpc2/collect.py task=walker_walk sb3_algo=sac steps=1000 morphology=True morphology_seed=1      # DMControl Env
```

Find the saved trajectory and the parsed version of it at the `outputs` directory.