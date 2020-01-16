from Network import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="splittingNetwork")
    parser.add_argument('--test_size',default=5,type=int)
    args = parser.parse_args()
    return args

if __name__ =='__main__':
    args=parse_args()
    test_size = args.test_size

    main()
