import argparse
from base_tools import *
from baseline import *
from any_folder import *


def test_any(model, data):
    """Testing"""
    model.eval()
    f_list = []
    name_list = []

    for imgname, img in data:
        img = img.cuda()
        with torch.no_grad():
            _, f = model(img)

        f_list.append(f.cpu().detach().numpy())
        name_list += list(imgname)

    return (name_list, f_list)


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config.device)
    seed_torch(seed=config.seed)

    # --- prepare data (Train Ref) ---
    ref_res_path = 'RefSet'
    makedirs(ref_res_path)
    train_ref_path = f'{config.root}/KADID_10k/ref_imgs'
    train_ref_path = [os.path.join(train_ref_path, 'I%02d.png' % (iii + 1)) for iii in list(range(81))]
    train_ref = torch.utils.data.DataLoader(AnyFolder(train_ref_path, '.png', config.patch_size),
                                            batch_size=1, shuffle=False)

    # --- prepare data (Additional Ref) ---
    additional_ref_path = f'{config.root}/KADID_10k/kadid_add81/ref_imgs'
    additional_ref = torch.utils.data.DataLoader(AnyFolder(additional_ref_path, '.png', config.patch_size),
                                                 batch_size=1, shuffle=False)

    # --- prepare model ---
    model = Baseline(pretrain=True, output_f=True).cuda()

    #
    train_ref_res = test_any(model, train_ref)
    additional_ref_res = test_any(model, additional_ref)

    dump_pkl(os.path.join(ref_res_path, 'train_ref.pkl'), train_ref_res)
    dump_pkl(os.path.join(ref_res_path, 'additional_ref.pkl'), additional_ref_res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', dest='device', type=int, default=0, help='0, 1, 2, 3')
    parser.add_argument('--seed', dest='seed', type=int, default=123, help='random seed')
    parser.add_argument('--root', dest='root', type=str, default='', help='IQA database root path')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                        help='Crop size for training & testing image patches')

    config = parser.parse_args()
    main(config)
