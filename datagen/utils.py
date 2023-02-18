import torch as t
import argparse 
import os

# Functions
def parse():
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('--data',           type=str,           required=True,                      help='Data type')    
    parser.add_argument('--bias_ratio',     type=str,           required=True,                      help='Minority ratio (e.g., 0.0,0.1,0.2,... (0.0 = without minority ,fully biased)')

    parser.add_argument('--data_storage',   type=str,           default='./storage/',               help='Raw data directory')
    parser.add_argument('--save_dir',       type=str,           default='../trainer/dataset/',      help='Where to store preprocessed dataset')
    
    args, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        raise SystemExit('Unknown parameters... : {}'.format(unknown))



    # Directory setting
    args.data_storage += args.data+'/'
    args.save_dir += args.data+'/'
    dir_gen(args.data_storage)
    dir_gen(args.save_dir)

    # Ratio, Noise setting
    args.bias_ratio = [float(b) for b in args.bias_ratio.split(',')]
    
    return args


def dir_gen(directory):
    try:
        os.makedirs(directory)
    except:
        pass



def save_data(data,directory):
    t.save(data,directory+'.pt')