import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--setting', type=str, choices= ['centralized', 'federated'], required= True, help='run mode')
    parser.add_argument('--algorithm', type=str, default = 'FedAvg', help='algorithm using in federated setting')
    parser.add_argument('--dataset', type=str, choices=['idda', 'femnist', 'gta5'], required=True, help='dataset name')
    parser.add_argument('--dataset2', type=str, choices=['idda', 'femnist', 'gta5'], required=True, help='dataset name')
    parser.add_argument('--niid', action='store_true', default=False,
                        help='Run the experiment with the non-IID partition (IID by default). Only on FEMNIST dataset.')
    parser.add_argument('--model', type=str, choices=['deeplabv3_mobilenetv2', 'resnet18', 'cnn'], help='model name')
    parser.add_argument('--num_rounds', type=int, help='number of rounds')
    parser.add_argument('--num_epochs', type=int, help='number of local epochs')
    parser.add_argument('--clients_per_round', type=int, help='number of clients trained per round')
    parser.add_argument('--hnm', action='store_true', default=False, help='Use hard negative mining reduction or not')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--bs', type=int, default=4, help='batch size')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--m', type=float, default=0.9, help='momentum')
    parser.add_argument('--print_train_interval', type=int, default=10, help='client print train interval')
    parser.add_argument('--print_test_interval', type=int, default=10, help='client print test interval')
    parser.add_argument('--eval_interval', type=int, default=10, help='eval interval')
    parser.add_argument('--test_interval', type=int, default=10, help='test interval')

    # ||| Transform Options |||
    parser.add_argument('--min_scale', type=float, default=0.25, help='define the lowest value for scale')
    parser.add_argument('--max_scale', type=float, default=2.0, help='define the highest value for scale')
    parser.add_argument('--h_resize', type=int, default=512, help='define the resize value for image H ')
    parser.add_argument('--w_resize', type=int, default=1024, help='define the resize value for image W ')
    parser.add_argument('--use_test_resize', action='store_true', default=False, help='whether to use test resize')
    parser.add_argument('--jitter', action='store_true', default=False, help='whether to use color jitter')
    parser.add_argument('--cv2_transform', action='store_true', default=False, help='whether to use cv2_transforms')
    parser.add_argument('--rrc_transform', action='store_true', default=False,
                        help='whether to use random resized crop')
    parser.add_argument('--rsrc_transform', action='store_true', default=False,
                        help='whether to use random scale random crop')
    parser.add_argument('--cts_norm', action='store_true', default=False,
                        help='whether to use cts normalization otherwise 0.5 for mean and std')
    

    return parser
