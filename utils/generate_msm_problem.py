from argparse import ArgumentParser
from random import randint, seed

from common import chunked_hex
from generate_ec_point import m, r, random_point


def parse_args():
    parser = ArgumentParser(description="Generate a BN254 MSM input file.")
    parser.add_argument("log_len", type=int)
    parser.add_argument("output")
    parser.add_argument(
        "--point-pool-log-len",
        type=int,
        default=12,
        help="generate this many random points and reuse them cyclically; scalars remain random for every row",
    )
    parser.add_argument("--seed", type=int, help="optional deterministic seed")
    return parser.parse_args()


def make_point_pool(pool_len: int):
    points = []
    for _ in range(pool_len):
        x, y = random_point()
        x_mont = (x * r) % m
        y_mont = (y * r) % m
        points.append(f"{chunked_hex(x_mont, 8)} {chunked_hex(y_mont, 8)}")
    return points


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        seed(args.seed)

    msm_len = 2 ** args.log_len
    point_pool_len = 2 ** min(args.log_len, args.point_pool_log_len)
    point_pool = make_point_pool(point_pool_len)

    with open(args.output, "w", encoding="utf-8") as wf:
        wf.write(f"{msm_len}\n")
        for i in range(msm_len):
            s = randint(0, m - 1)
            wf.write(f"{chunked_hex(s, 8)}|{point_pool[i % point_pool_len]}\n")
