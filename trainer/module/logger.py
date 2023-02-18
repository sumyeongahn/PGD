__author__ = 'Anonymizing'
__email__ = 'Anonymizing'

from datetime import datetime
import logging

def level_info(log_level):
    if log_level == 'debug':
        return logging.DEBUG
    elif log_level == 'info':
        return logging.INFO
    elif log_level == 'warning':
        return logging.WARNING
    elif log_level == 'error':
        return logging.ERROR
    elif log_level == 'critical':
        return logging.CRITICAL

def logger(args):
    __logger = logging.getLogger('Debiasing feature bias with noisy labels')
    __logger.setLevel(level_info('debug'))
    formatter = logging.Formatter('(%(asctime)s): %(message)s')

    # Stream handler
    sthandler = logging.StreamHandler()
    sthandler.setFormatter(formatter)
    sthandler.setLevel(level_info('critical'))

    # File hanlder
    mode = 'w'
    now = datetime.now()
    current_time = 'debug' if 'debug' in args.exp else '%s-%s-%s-%s-%s-%s' %(now.year,now.month,now.day,now.hour,now.minute,now.second)
    if args.train:
        fhandler = logging.FileHandler(args.log_dir+current_time+'.log', mode=mode)    
    else:
        fhandler = logging.FileHandler('/dev/null', mode=mode)    
    fhandler.setFormatter(formatter)
    fhandler.setLevel(level_info('debug'))
    

    __logger.addHandler(sthandler)
    __logger.addHandler(fhandler)

    return __logger
        
