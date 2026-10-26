import numpy as np
import argparse
import dataset
import json
import metrics
import methods.herl
from methods import *
import os
import sys
import methods.util as util
import warnings

warnings.filterwarnings('ignore')
def round_4(obj):
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return round(float(obj.item()), 5)
        return np.round(obj.detach().cpu().numpy(), 5).tolist()

    if isinstance(obj, np.ndarray):
        return np.round(obj, 5).tolist()

    if isinstance(obj, (np.float32, np.float64, float)):
        return round(float(obj), 5)

    if isinstance(obj, (np.int32, np.int64, int)):
        return int(obj)

    if isinstance(obj, dict):
        return {k: round_4(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [round_4(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(round_4(v) for v in obj)

    return obj
def training_main():
    tot_epoch = config['epoch_num']
    flag = 1
    for epoch in range(mth.epoch, tot_epoch):
        sys.stdout.flush()
        losses = mth.train_epoch(flag)
        if epoch >= 150:
            flag = 0
        if epoch % 1 == 0:
            save_everything(f'ckpt{epoch}')

        if (epoch % 15 == 0) or (epoch == tot_epoch - 1) or epoch >= 140:
            fused_score, zscore_ener, zscore_reer, score_test_ener, score_test_reer, thresh, pred = mth.knownpred_unknwonscore_test(
                test_loader)
            acc = evaluation.close_accuracy(pred)
            open_detection = evaluation.open_detection_indexes(fused_score, thresh)
            fused_score_auroc = open_detection['auroc']
            zscore_ener_auroc = evaluation.open_detection_indexes(zscore_ener, thresh)['auroc']
            zscore_reer_auroc = evaluation.open_detection_indexes(zscore_reer, thresh)['auroc']
            score_test_ener_auroc = evaluation.open_detection_indexes(score_test_ener, thresh)['auroc']
            score_test_reer_auroc = evaluation.open_detection_indexes(score_test_reer, thresh)['auroc']

            history_item = {
                "loss": losses,
                "close acc": acc,
                "fused_score_auroc": fused_score_auroc,
                "zscore_ener_auroc": zscore_ener_auroc,
                "zscore_reer_auroc": zscore_reer_auroc,
                "score_test_ener_auroc": score_test_ener_auroc,
                "score_test_reer_auroc": score_test_reer_auroc,
                "open_detection": open_detection,
                "open_reco": evaluation.open_reco_indexes(fused_score, thresh, pred),
            }

            log_history(epoch, round_4(history_item))

        else:
            log_history(epoch, {
                "loss": losses,
            })
        save_everything()

import json
import torch
import numpy as np

def make_json_serializable(obj):
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.item()
        else:
            return obj.detach().cpu().tolist()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)

    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)

    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(make_json_serializable(v) for v in obj)

    return obj

def save_everything(subfix=""):
    if subfix == "":
        mth.save_model(saving_path + 'model.pth')
    with open(saving_path + "history.json", 'w') as f:
        history_json = make_json_serializable(history)
        json.dump(history_json, f, indent=4)

def log_history(epoch, data_dict):
    item = {
        'epoch': epoch
    }
    item.update(data_dict)
    if isinstance(history, list):
        history.append(item)
    print(f"Epoch:{epoch}")
    for key in data_dict.keys():
        print("  ", key, ":", data_dict[key])
def update_config_keyvalues(config, update):
    if update == "":
        return config
    spls = update.split(",")
    for spl in spls:
        key, val = spl.split(':')
        key_parts = key.split('.')
        sconfig = config
        for i in range(len(key_parts) - 1):
            sconfig = sconfig[key_parts[i]]
        org = sconfig[key_parts[-1]]
        if isinstance(org, bool):
            sconfig[key_parts[-1]] = val == 'True'
        elif isinstance(org, int):
            sconfig[key_parts[-1]] = int(val)
        elif isinstance(org, float):
            sconfig[key_parts[-1]] = float(val)
        else:
            sconfig[key_parts[-1]] = val
        print("Updating", key, "with", val, "results in", sconfig[key_parts[-1]])
    return config


def update_subconfig(cfg, u):
    for k in u.keys():
        if not k in cfg.keys() or not isinstance(cfg[k], dict):
            cfg[k] = u[k]
        else:
            update_subconfig(cfg[k], u[k])


def load_config(file):
    with open(file, "r") as f:
        config = json.load(f)
    if 'inherit' in config.keys():
        inheritfile = config['inherit']
        if inheritfile != 'None':
            parent = load_config(inheritfile)
            update_subconfig(parent, config)
            config = parent
    return config

def set_up_gpu(args):
    if args.gpu != 'cpu':
        args.gpu = ",".join([c for c in args.gpu])
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


if __name__ == "__main__":
    datasetName='cifar10'
    for R in [50]:
        for x in ['a']:
            print(str(x) + "    --------------------------------  " + str(x))
            import torch

            torch.backends.cudnn.benchmark = True
            parser = argparse.ArgumentParser()
            parser.add_argument('--method', type=str, required=False, default="herl",
                                help='Methods : ' + ",".join(util.method_list.keys()))
            parser.add_argument('--test', action="store_true", help='Evaluation mode')
            parser.add_argument('--configupdate', type=str, required=False, default="",
                                help='Update several key values in config')
            parser.add_argument('--test_interval', type=int, required=False, default=1,
                                help='The frequency of model evaluation')
            parser.add_argument('--gpu', type=str, required=False, default="0", help='GPU number')
            parser.add_argument('--ds', type=str, required=False,
                                default="./split/" + datasetName + "/spl_" + x + ".json",
                                help='dataset setting, choose file from ./exps')
            parser.add_argument('--config', type=str, required=False,
                                default="./configs/" + datasetName + ".json",
                                help='model configuration, choose from ./configs')
            parser.add_argument('--save', type=str, required=False, default="/" + datasetName + "/" + x + "_m5",
                                help='Saving folder name')

            args = parser.parse_args()
            test_interval = args.test_interval
            if not args.save.endswith("/"):
                args.save += "/"
            set_up_gpu(args)
            if args.config != "None":
                config = load_config(args.config)
            else:
                config = {}
            config["R_c"] = R
            config = update_config_keyvalues(config, args.configupdate)
            args.bs = config['batch_size']
            saving_path = "./save/" + args.save
            util.setup_dir(saving_path)
            saving_path = saving_path + "_" + str(R)
            train_loader, test_loader, classnum = dataset.load_partitioned_dataset(args, args.ds)
            mth = util.method_list[args.method](config, classnum, train_loader.dataset)

            history = []
            evaluation = metrics.OSREvaluation(test_loader)

            if not args.test:
                print(f"TotalEpochs:{config['epoch_num']}")
                training_main()
                save_everything()






