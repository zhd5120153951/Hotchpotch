import argparse
from pixel_init import *


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', nargs='+', default=['SVC'], help='SVC')
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    args = parse_opt()

    model_name = args.model_name[0]

    #job 1 train the pixels
    # model_train(model_name)
    #job 2 test the job2's model
    model_predict(model_name)