from generate_ec_point import *
from random import randint
from sys import argv

if __name__ == "__main__":
    msm_len = 2 ** int(argv[1])

    curve = CURVE_MNT4753

    wf = open(argv[2], 'w')

    wf.write(f"{msm_len}\n")

    for i in range(msm_len):
        s = randint(0, 0x01C4C62D92C41110229022EEE2CDADB7F997505B8FAFED5EB7E8F96C97D87307FDB925E8A0ED8D99D124D9A15AF79DB26C5C28C859A99B3EEBCA9429212636B9DFF97634993AA4D6C381BC3F0057974EA099170FA13A4FD90776E240000001 - 1)
        x, y = curve.random_point()
        x_mont = (x * curve.r) % curve.m
        y_mont = (y * curve.r) % curve.m
        wf.write(f'{chunked_hex(s, 24)}|{chunked_hex(x_mont, 24)} {chunked_hex(y_mont, 24)}\n')
    
    wf.close()
        
