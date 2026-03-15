from args.args import parse_args
from train import run


def main():
    args = parse_args()
    args.test = True
    run(args)


if __name__ == '__main__':
    main()
