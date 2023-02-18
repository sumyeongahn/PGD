
def reproduce(args):
    args.model = 'CONV'
    args.opt = 'SGD'
    args.batch_size = 128
    args.lr = 0.02
    args.n_lr = 0.02
    args.weight_decay = 0.001
    args.momentum = 0.9
    args.num_class = 10
    args.use_lr_decay=True
    args.lr_decay = 0.1
    args.lr_decay_step = 40
    return args

