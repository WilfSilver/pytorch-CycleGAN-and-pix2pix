import itertools
import os
import subprocess
import sys
from typing import Iterable


base_params = [
    # Put all checkpoints on shared directory
    "--checkpoints_dir", "~/checkpoints",
    "--dataroot", "./datasets/facades",
    "--model", "pix2pix",
    "--direction", "BtoA",
    "--dataset_mode", "aligned",
    "--norm", "batch",
]

train_default_params = base_params + [
    "--pool_size", "0",
    "--no_html",
    "--save_epoch_freq", "300",
    "--use_wandb"
]

test_default_params = base_params + [
    "--results_dir", "~/results",
]

# 2: --max_dataset_size 200 400
# 2: --lr_policy linear plateu
# 2: --lr 0.0002 0.0003
# 3: --lambda_L1 50 100 150
# 3: --lambda_perceptual 5 10 15
# 2: --netG unet_256 resnet_6blocks
params_to_test = {
    "--netG": ["unet_256", "resnet_9blocks"],

    "--max_dataset_size": [200, 400],
    "--lr_policy": ["linear", "plateau"],
    "--lr": [0.0002, 0.0003],
    "--lambda_L1": [50, 100, 150],
    "--lambda_perceptual": [5, 10, 15],
}

def gen_possibilities(params_to_test: dict[str, list[str | int]]) -> list[list[str]]:
    def get_possibility(params: Iterable[str], i: int):
        values = []
        for p in params:
            m = len(params_to_test[p])
            values.append(params_to_test[p][i % m])
            i //= m

        if i > 0:
            return None

        return itertools.chain.from_iterable(zip(params, values))

    all_possibilities = []
    params = params_to_test.keys()
    i = 0
    while (v := get_possibility(params, i)) is not None:
        all_possibilities.append(list(map(str, v)))
        i += 1

    return all_possibilities


def get_name(opts: list[str]):
    return "_".join((opts[i] for i in range(1, len(opts), 2)))



all_possibilities = gen_possibilities(params_to_test)

num_devices = 10
device = int(sys.argv[1])
env = dict(os.environ, CUDA=sys.argv[2])
for i in range(device, len(all_possibilities), num_devices):
    opts = all_possibilities[i]
    name = get_name(opts)

    cmd = ["python", "train.py"] + train_default_params + opts + ["--name", name]
    print("! " + " ".join(cmd))
    subprocess.run(cmd, env=env)

    netg_i = opts.index("--netG") + 1

    cmd = ["python", "test.py"] + test_default_params + ["--name", name, "--netG", opts[netg_i]]
    print("! " + " ".join(cmd))
    res = subprocess.run(cmd, env=env, capture_output=True)

    lines = res.stdout.splitlines()
    for line in lines[::-1]:
        if len(line) > 0:
            last_line = line.decode()
            break

    print(last_line)
    with open("~/res.txt", "a") as file:
        file.write(f"{name},{last_line}\n")

