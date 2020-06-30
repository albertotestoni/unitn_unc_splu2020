import argparse
import csv
import datetime
import json
import sys
from time import time

import numpy as np
import os
import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models.LXMERTOracleInputTarget import LXMERTOracleInputTarget
from utils.config import load_config
from utils.datasets.Oracle.LXMERTOracleDataset import LXMERTOracleDataset
from utils.model_loading import load_model
from utils.vocab import create_vocab


def calculate_accuracy_oracle(predictions, targets):
    """
    :param prediction: NxC
    :param targets: N
    """
    if isinstance(predictions, Variable):
        predictions = predictions.data
    if isinstance(targets, Variable):
        targets = targets.data

    predicted_classes = predictions.topk(1)[1]
    accuracy = torch.eq(predicted_classes.squeeze(1), targets).sum().item()/targets.size(0)
    return accuracy

def calculate_accuracy_oracle_all(predictions, targets):
    """
    :param prediction: NxC
    :param targets: N
    """
    if isinstance(predictions, Variable):
        predictions = predictions.data
    if isinstance(targets, Variable):
        targets = targets.data

    accuracies = []
    predicted_classes = predictions.topk(1)[1]
    for accuracy in torch.eq(predicted_classes.squeeze(1), targets):
        accuracies.append(accuracy.item())
    return accuracies

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-config", type=str, default="config/Oracle/config_small.json", help='Config file')
    parser.add_argument("-img_feat", type=str, default="vgg", help='Select "vgg" or "res" as image features')
    parser.add_argument("-exp_name", type=str, help='Experiment Name')
    parser.add_argument("-bin_name", type=str, default='', help='Name of the trained model file')
    parser.add_argument("--preloaded", type=bool, default=True)
    parser.add_argument("-load_bin_path", type=str)

    args = parser.parse_args()

    config = load_config(args.config)

    # Experiment Settings
    exp_config = config['exp_config']
    exp_config['img_feat'] = args.img_feat.lower()
    exp_config['use_cuda'] = torch.cuda.is_available()
    exp_config['ts'] = str(datetime.datetime.fromtimestamp(time()).strftime('%Y_%m_%d_%H_%M'))

    torch.manual_seed(exp_config['seed'])
    if exp_config['use_cuda']:
        torch.cuda.manual_seed_all(exp_config['seed'])
        
    if exp_config['logging']:
        exp_config['name'] = args.exp_name
        if not os.path.exists(exp_config["tb_logdir"] + "oracle_" + exp_config["name"]):
            os.makedirs(exp_config["tb_logdir"] + "oracle_" + exp_config["name"])
        writer = SummaryWriter(exp_config["tb_logdir"] + "oracle_" + exp_config["name"])
        train_batch_out = 0
        valid_batch_out = 0

    # Hyperparamters
    data_paths          = config['data_paths']
    optimizer_config    = config['optimizer']
    embedding_config    = config['embeddings']
    lstm_config         = config['lstm']
    mlp_config          = config['mlp']
    dataset_config      = config['dataset']
    inputs_config       = config['inputs']

    print("Loading MSCOCO bottomup index from: {}".format(data_paths["FasterRCNN"]["mscoco_bottomup_index"]))
    with open(data_paths["FasterRCNN"]["mscoco_bottomup_index"]) as in_file:
        mscoco_bottomup_index = json.load(in_file)
        image_id2image_pos = mscoco_bottomup_index["image_id2image_pos"]
        image_pos2image_id = mscoco_bottomup_index["image_pos2image_id"]
        img_h = mscoco_bottomup_index["img_h"]
        img_w = mscoco_bottomup_index["img_w"]

    print("Loading MSCOCO bottomup features from: {}".format(data_paths["FasterRCNN"]["mscoco_bottomup_features"]))
    mscoco_bottomup_features = None
    if args.preloaded:
        import sharearray
        print("Loading preloaded MS-COCO Bottom-Up features")
        mscoco_bottomup_features = sharearray.cache("mscoco_vectorized_features", lambda: None)
        mscoco_bottomup_features = np.array(mscoco_bottomup_features)
    else:
        mscoco_bottomup_features = np.load(data_paths["FasterRCNN"]["mscoco_bottomup_features"])

    print("Loading MSCOCO bottomup boxes from: {}".format(data_paths["FasterRCNN"]["mscoco_bottomup_boxes"]))
    mscoco_bottomup_boxes = None
    if args.preloaded:
        print("Loading preloaded MS-COCO Bottom-Up boxes")
        mscoco_bottomup_boxes = sharearray.cache("mscoco_vectorized_boxes", lambda: None)
        mscoco_bottomup_boxes = np.array(mscoco_bottomup_boxes)
    else:
        mscoco_bottomup_boxes = np.load(data_paths["FasterRCNN"]["mscoco_bottomup_boxes"])

    imgid2fasterRCNNfeatures = {}
    for mscoco_id, mscoco_pos in image_id2image_pos.items():
        imgid2fasterRCNNfeatures[mscoco_id] = dict()
        imgid2fasterRCNNfeatures[mscoco_id]["features"] = mscoco_bottomup_features[mscoco_pos]
        imgid2fasterRCNNfeatures[mscoco_id]["boxes"] = mscoco_bottomup_boxes[mscoco_pos]
        imgid2fasterRCNNfeatures[mscoco_id]["img_h"] = img_h[mscoco_pos]
        imgid2fasterRCNNfeatures[mscoco_id]["img_w"] = img_w[mscoco_pos]

    if dataset_config['new_vocab'] or not os.path.isfile(os.path.join(args.data_dir, data_paths['vocab_file'])):
        create_vocab(
            data_dir=args.data_dir,
            data_file=data_paths['train_file'],
            min_occ=dataset_config['min_occ'])

    print("Bert tokenizer or standard vocab?")

    with open(os.path.join(args.data_dir, data_paths['vocab_file'])) as file:
        vocab = json.load(file)

    # with open('./data/vocab_bert_tok.json') as file:
    #     vocab = json.load(file)

    word2i = vocab['word2i']
    i2word = vocab['i2word']
    vocab_size = len(word2i)

    # Init Model, Loss Function and Optimizer
    model = LXMERTOracleInputTarget(
        no_words            = vocab_size,
        no_words_feat       = embedding_config['no_words_feat'],
        no_categories       = embedding_config['no_categories'],
        no_category_feat    = embedding_config['no_category_feat'],
        no_hidden_encoder   = lstm_config['no_hidden_encoder'],
        mlp_layer_sizes     = mlp_config['layer_sizes'],
        no_visual_feat      = inputs_config['no_visual_feat'],
        no_crop_feat        = inputs_config['no_crop_feat'],
        dropout             = lstm_config['dropout'],
        inputs_config       = inputs_config,
        scale_visual_to     = inputs_config['scale_visual_to'],
        lxmert_encoder_args = inputs_config["LXRTEncoder"]
    )
    model = load_model(model, args.load_bin_path, use_dataparallel=exp_config["use_cuda"])

    dataset_test = LXMERTOracleDataset(
        data_dir            = args.data_dir,
        data_file           = data_paths['test_file'],
        split               = 'test',
        visual_feat_file    = data_paths[args.img_feat]['image_features'],
        visual_feat_mapping_file = data_paths[exp_config['img_feat']]['img2id'],
        visual_feat_crop_file = data_paths[args.img_feat]['crop_features'],
        visual_feat_crop_mapping_file = data_paths[exp_config['img_feat']]['crop2id'],
        max_src_length      = dataset_config['max_src_length'],
        hdf5_visual_feat    = 'test_img_features',
        hdf5_crop_feat      = 'crop_features',
        imgid2fasterRCNNfeatures = imgid2fasterRCNNfeatures,
        history             = dataset_config['history'],
        new_oracle_data     = True, #dataset_config['new_oracle_data']
        successful_only     = dataset_config['successful_only'],
        load_crops=True,
        only_location=False
    )

    accuracy = []

    dataloader = DataLoader(
        dataset=dataset_test,
        batch_size=optimizer_config['batch_size'],
        shuffle=False,
        num_workers=0 if sys.gettrace() else 4,
        pin_memory=exp_config['use_cuda']
    )

    torch.set_grad_enabled(False)
    model.eval()

    ans2tok = {'Yes': 1,
               'No': 0,
               'N/A': 2}

    tok2ans = {v: k for k, v in ans2tok.items()}

    pos = 0
    last_game_id = None
    with open("lxmert_scratch_small_predictions.csv", mode="w") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["Game ID", "Position", "Image", "Question", "GT Answer", "Model Answer"])
        stream = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)
        for i_batch, sample in stream:
            # Get Batch
            questions, answers, crop_features, visual_features, spatials, obj_categories, lengths = \
                sample['question'], sample['answer'], sample['crop_features'], sample['img_features'], sample['spatial'], sample['obj_cat'], sample['length']

            # Forward pass
            pred_answer = model(Variable(questions),
                Variable(obj_categories),
                Variable(spatials),
                Variable(crop_features),
                Variable(visual_features),
                Variable(lengths),
                sample["history_raw"],
                sample['FasterRCNN']['features'],
                sample['FasterRCNN']['boxes'],
                sample["target_bbox"]
            )

            # Calculate Accuracy
            accuracy.extend(calculate_accuracy_oracle_all(pred_answer, answers.cuda() if exp_config['use_cuda'] else answers))

            stream.set_description("Accuracy: {}".format(np.round(np.mean(accuracy), 2)))
            stream.refresh()  # to show immediately the update

            if i_batch == 0:
                last_game_id = sample["game_id"][0]

            pred_answer_topk = pred_answer.topk(1)[1]

            for i in range(questions.shape[0]):
                question_raw = " ".join([i2word[str(idx.item())] for idx in questions[i] if idx.item() != 0])
                answer_raw = tok2ans[answers[i].item()]
                pred_answer_raw = tok2ans[pred_answer_topk[i].item()]
                game_id = sample["game_id"][i]

                if last_game_id != game_id:
                    last_game_id = game_id
                    pos = 0

                writer.writerow(
                    [
                        game_id,
                        pos,
                        sample["image_file"][i],
                        question_raw,
                        answer_raw,
                        pred_answer_raw
                    ]
                )

                pos += 1

        print("Test accuracy: {}".format(np.mean(accuracy)))
