import datetime
import argparse
import time

from BaseIQASolver import BaseIQASolver
import data_loader
from base_tools import *


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config.device)
    seed_torch(seed=config.seed)
    dir_init(config)

    experiment_name = '{}_{}_cross_{}_{}'.format(datetime.datetime.now().strftime('%m-%d_%H_%M_%S'), config.dataset,
                                                 config.test_db, config.description)
    Logger_init(os.path.join(config.logging_dir, '{}.txt'.format(experiment_name)))

    if config.model:
        model_root = os.path.join(config.model_dir, experiment_name)
        makedirs(model_root)

    print(experiment_name)
    print(config)
    print(f'Epoch\t{config.test_db.upper()}_SRCC\t{config.test_db.upper()}_PLCC\tTrain_Loss\tTrain_SRCC\tCost_Time')

    best_srcc, best_plcc = 0., 0.
    start_time = time.time()

    solver = BaseIQASolver(config)
    ori_train_data = gen_data(config.root, config.dataset, config.patch_size, config.train_patch_num, config.batch_size,
                              False, test_db=config.test_db)
    test_data = gen_data(config.root, config.test_db, config.patch_size, config.test_patch_num, 1, False)

    for t in range(config.epochs):
        filtered_idx = solver.downsample(ori_train_data)
        train_data = gen_data(config.root, config.dataset, config.patch_size, config.train_patch_num,
                              config.batch_size, True, filtered_idx, test_db=config.test_db)

        epoch_loss, train_srcc = solver.train(train_data)

        if_best = False
        test_srcc, test_plcc, test_res = solver.test(test_data)
        if test_srcc > best_srcc:
            best_srcc, best_plcc = test_srcc, test_plcc
            if_best = True
            print('*', end='')

        print('%d\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.2f' % (
            t, test_srcc, test_plcc, sum(epoch_loss) / len(epoch_loss), train_srcc, time.time() - start_time))

        if config.model:
            save_checkpoint({
                'epoch': t,
                'state_dict': solver.model.state_dict(),
                'best_SRCC': best_srcc,
                'best_PLCC': best_plcc,
                'optimizer': solver.solver.state_dict()},
                if_best,
                os.path.join(model_root, '{}_checkpoint.pth.tar'.format(config.dataset)),
                os.path.join(model_root, '{}_model_best.pth.tar'.format(config.dataset))
            )

        start_time = time.time()

    print('Best cross SRCC:', np.round(best_srcc, 4))
    print('Best cross PLCC:', np.round(best_plcc, 4))


def gen_data(root, dataset, patch_size, patch_num, batch_size, istrain, filtered_idx=None, test_db=None):
    folder_path, sel_num = get_db_base_info(root, dataset)
    data = data_loader.DataLoader(dataset, folder_path, sel_num, patch_size,
                                  patch_num, batch_size=batch_size,
                                  istrain=istrain, filtered_idx=filtered_idx, test_db=test_db).get_data()

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiment description
    parser.add_argument('--description', dest='description', type=str, default='test',
                        help='the description of the experiment')
    # base config
    parser.add_argument('--device', dest='device', type=int, default=0, help='0, 1, 2, 3')
    parser.add_argument('--dataset', dest='dataset', type=str, default='kadid-10k',
                        help='Support datasets: kadid-10k')
    parser.add_argument('--test_db', dest='test_db', type=str, default='bid',
                        help='target testing database, e.g. bid|livec|koniq-10k')
    parser.add_argument('--seed', dest='seed', type=int, default=123, help='random seed')
    parser.add_argument('--root', dest='root', type=str, default='', help='IQA database root path')

    # network config
    parser.add_argument('--pretrain', dest='pretrain', action='store_false',
                        help='whether to pretrain in ImageNet')

    # hyper-parameters config
    parser.add_argument('--epochs', dest='epochs', type=int, default=24, help='Epochs for training')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                        help='Crop size for training & testing image patches')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=1,
                        help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=5,
                        help='Number of sample patches from testing image')

    parser.add_argument('--lr', dest='lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10,
                        help='Learning rate ratio for hyper network')

    # saving config
    parser.add_argument('--saving_model', dest='model', action='store_true',
                        help='whether to save models')

    # path config
    parser.add_argument('--logging_dir', dest='logging_dir', type=str, default='./logging',
                        help='path for saving logging')
    parser.add_argument('--model_dir', dest='model_dir', type=str, default='./model',
                        help='path for saving models')
    parser.add_argument('--feature_dir', dest='feature_dir', type=str, default='./feature',
                        help='path for saving features')

    config = parser.parse_args()
    main(config)
