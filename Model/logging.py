import os
import torchvision


def launch(dir):
    os.system('tensorboard --logdir ' + dir + ' --port=6006')
    return


def rm_old_log(dir='logs'):
    ##launch tensorboard automatically
    os.system('lsof -i tcp:6006 | grep -v PID | awk \'{print $2}\' | xargs kill')
    os.system('rm -r ' + dir + '/train')
    os.system('rm -r ' + dir + '/valid')


def launchTensorBoard(dir='logs', rm_old=True):
    import threading
    if rm_old:
        rm_old_log()
    t = threading.Thread(target=launch, args=([dir]))
    t.start()
    return t


def train_log_action(logger, image, pred, loss, global_step):
    '''
    this function changes depending on logging task
    '''
    logger.add_scalar('loss', loss, global_step)
    if global_step % 20 == 0:  ## store image after 20 global steps
        logger.add_image('image', torchvision.utils.make_grid(image, pad_value=1), global_step)


def val_log_action(logger, image, pred, loss, global_step):
    '''
    this function changes depending on logging task
    '''
    logger.add_scalar('loss', loss, global_step)
    logger.add_image('image', torchvision.utils.make_grid(image, pad_value=1), global_step)
