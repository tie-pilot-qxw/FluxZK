from typing import Optional, Tuple
from random import randint
from pathlib import Path
from dataclasses import dataclass
from math import log2

from common import *

@dataclass
class Curve:
    m: int
    k: int
    a: int
    b: int
    r: int

    @staticmethod
    def new(m: int, r: int, a: int, b: int) -> Curve:
        """
        Create a curve on F_m (represented with log_2 r bits)
        """
        s = 1
        while ((m - 1) // s) % 2 == 0:
            s *= 2
        print(f"s = {int(log2(s))}")
        k = ((m - 1) // s + 1) // 2
        return Curve(m=m, r=r, a=a, b=b, k=k)

    def try_sqrt(self, x: int) -> Optional[int]:
        x_sqrt = quick_pow(x, self.m, self.k)
        if (x_sqrt * x_sqrt) % self.m == x:
            return x_sqrt
        return None

    def random_point(self) -> Tuple[int, int]:
        while True:
            x = randint(0, self.m - 1)
            y2 = (x ** 3) % self.m + self.a * x + self.b
            y2 = y2 % self.m
            y = self.try_sqrt(y2)

            if y != None:
                if randint(0, 1) == 0:
                    return x, y
                else:
                    return x, (-y) % self.m

CURVE_BN254 = Curve.new(21888242871839275222246405745257275088696311157297823662689037894645226208583, 2 ** 256, 0, 3)
CURVE_BLS12_381 = Curve.new(0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab, 2 ** 384, 0, 4)
CURVE_MNT4753 = Curve.new(0x01C4C62D92C41110229022EEE2CDADB7F997505B8FAFED5EB7E8F96C97D87307FDB925E8A0ED8D99D124D9A15AF79DB117E776F218059DB80F0DA5CB537E38685ACCE9767254A4638810719AC425F0E39D54522CDD119F5E9063DE245E8001,
2** 768, 2, 0x01373684A8C9DCAE7A016AC5D7748D3313CD8E39051C596560835DF0C9E50A5B59B882A92C78DC537E51A16703EC9855C77FC3D8BB21C8D68BB8CFB9DB4B8C8FBA773111C36C8B1B4E8F1ECE940EF9EAAD265458E06372009C9A0491678EF4)

if __name__ == "__main__":
    c = CURVE_BN254
    x, y = c.random_point()
    # Convert to Montgomery representation
    x_mont = (x * c.r) % c.m
    y_mont = (y * c.r) % c.m
    print(f"// x = {x}")
    print(f"// x_mont = {x_mont}")
    print(f"const u32 x[8] = BIG_INTEGER_CHUNKS8({chunks(x_mont, 8)});")
    print(f"// y = {y}")
    print(f"// y_mont = {y_mont}")
    print(f"const u32 y[8] = BIG_INTEGER_CHUNKS8({chunks(y_mont, 8)});")

