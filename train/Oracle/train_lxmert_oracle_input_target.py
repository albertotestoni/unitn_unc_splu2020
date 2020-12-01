import argparse
import datetime
import json
import os
import sys
from collections import OrderedDict
from time import time

import numpy as np
import torch
import torch.nn as nn
import tqdm
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from lxmert.src.lxrt.optimization import BertAdam
from models.LXMERTOracleInputTarget import LXMERTOracleInputTarget
from utils.config import load_config
from utils.datasets.Oracle.LXMERTOracleDataset import LXMERTOracleDataset
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
    accuracy = torch.eq(predicted_classes.squeeze(1), targets).sum().item() / targets.size(0)
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-config", type=str, default="config/Oracle/config.json", help='Config file')
    parser.add_argument("-img_feat", type=str, default="vgg", help='Select "vgg" or "res" as image features')
    parser.add_argument("-exp_name", type=str, help='Experiment Name')
    parser.add_argument("-bin_name", type=str, default='', help='Name of the trained model file')
    parser.add_argument("--preloaded", type=bool, default=False)

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
    data_paths = config['data_paths']
    optimizer_config = config['optimizer']
    embedding_config = config['embeddings']
    lstm_config = config['lstm']
    mlp_config = config['mlp']
    dataset_config = config['dataset']
    inputs_config = config['inputs']

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

    with open(os.path.join(args.data_dir, data_paths['vocab_file'])) as file:
        vocab = json.load(file)

    word2i = vocab['word2i']
    i2word = vocab['i2word']
    vocab_size = len(word2i)

    # Init Model, Loss Function and Optimizer
    model = LXMERTOracleInputTarget(
        no_categories=embedding_config['no_categories'],
        no_category_feat=embedding_config['no_category_feat'],
        no_hidden_encoder=lstm_config['no_hidden_encoder'],
        mlp_layer_sizes=mlp_config['layer_sizes'],
        no_visual_feat=inputs_config['no_visual_feat'],
        no_crop_feat=inputs_config['no_crop_feat'],
        inputs_config=inputs_config,
        scale_visual_to=inputs_config['scale_visual_to'],
        lxmert_encoder_args=inputs_config["LXRTEncoder"]
    )

    loss_function = nn.CrossEntropyLoss()

    if exp_config['use_cuda']:
        model.cuda()
        model = DataParallel(model)
        print(model)

    if exp_config['logging']:
        writer.add_text("Experiment Configuration", str(exp_config))
        writer.add_text("Model", str(model))

    dataset_train = LXMERTOracleDataset(
        data_dir=args.data_dir,
        data_file=data_paths['train_file'],
        split='train',
        visual_feat_file=data_paths[args.img_feat]['image_features'],
        visual_feat_mapping_file=data_paths[exp_config['img_feat']]['img2id'],
        visual_feat_crop_file=data_paths[args.img_feat]['crop_features'],
        max_src_length=dataset_config['max_src_length'],
        hdf5_visual_feat='train_img_features',
        hdf5_crop_feat='crop_features',
        imgid2fasterRCNNfeatures=imgid2fasterRCNNfeatures,
        history=dataset_config['history'],
        new_oracle_data=dataset_config['new_oracle_data'],
        successful_only=dataset_config['successful_only'],
        load_crops=True,
        only_location=False
    )

    dataset_validation = LXMERTOracleDataset(
        data_dir=args.data_dir,
        data_file=data_paths['val_file'],
        split='val',
        visual_feat_file=data_paths[args.img_feat]['image_features'],
        visual_feat_mapping_file=data_paths[exp_config['img_feat']]['img2id'],
        visual_feat_crop_file=data_paths[args.img_feat]['crop_features'],
        max_src_length=dataset_config['max_src_length'],
        hdf5_visual_feat='val_img_features',
        hdf5_crop_feat='crop_features',
        imgid2fasterRCNNfeatures=imgid2fasterRCNNfeatures,
        history=dataset_config['history'],
        new_oracle_data=dataset_config['new_oracle_data'],
        successful_only=dataset_config['successful_only'],
        load_crops=True,
        only_location=False
    )

    print("Initializing the optimizer...")
    num_batches_per_epoch = len(dataset_train) // optimizer_config['batch_size']
    num_total_batches = num_batches_per_epoch * optimizer_config['no_epochs']
    print("Number of batches per epoch: {}".format(num_batches_per_epoch))
    print("Total number of batches: {}".format(num_total_batches))
    optimizer = BertAdam(model.parameters(), lr=optimizer_config['lr'], warmup=0.1, t_total=num_total_batches)

    for epoch in range(optimizer_config['no_epochs']):
        start = time()
        loss, train_accuracy, val_accuracy = 0, 0, 0

        if exp_config['use_cuda']:
            train_loss = torch.cuda.FloatTensor()
            val_loss = torch.cuda.FloatTensor()
        else:
            train_loss = torch.FloatTensor()
            val_loss = torch.FloatTensor()

        for split, dataset in zip(exp_config['splits'], [dataset_train, dataset_validation]):
            accuracy = []

            dataloader = DataLoader(
                dataset=dataset,
                batch_size=optimizer_config['batch_size'],
                shuffle=True,
                num_workers=0 if sys.gettrace() else 4,
                pin_memory=exp_config['use_cuda']
            )
            if split == 'train':
                torch.set_grad_enabled(True)
                model.train()
            else:
                torch.set_grad_enabled(False)
                model.eval()

            stream = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)
            for i_batch, sample in stream:
                # Get Batch
                questions, answers, crop_features, visual_features, spatials, obj_categories, lengths, train_features = \
                        sample['question'], sample['answer'], sample['crop_features'], sample['img_features'], sample['spatial'], sample['obj_cat'], sample['length'], sample['train_features']

                # Forward pass
                pred_answer = model(
                    Variable(crop_features),
                    sample['FasterRCNN']['features'],
                    sample['FasterRCNN']['boxes'],
                    sample["target_bbox"],
                    Variable(sample['train_features']['input_ids']),
                    Variable(sample['train_features']['input_mask']),
                    Variable(sample['train_features']['segment_ids'])
                )

                # Calculate Loss
                loss = loss_function(pred_answer, Variable(answers).cuda() if exp_config['use_cuda'] else Variable(
                    answers)).unsqueeze(0)

                # Calculate Accuracy
                accuracy.append(
                    calculate_accuracy_oracle(pred_answer, answers.cuda() if exp_config['use_cuda'] else answers))

                stream.set_description("Train accuracy: {}".format(np.round(np.mean(accuracy), 2)))
                stream.refresh()

                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss = torch.cat([train_loss, loss.data])

                else:
                    val_loss = torch.cat([val_loss, loss.data])

                # bookkeeping
                if split == 'train' and exp_config['logging']:
                    writer.add_scalar("Training/Batch Accuracy", accuracy[-1], train_batch_out)
                    writer.add_scalar("Training/Batch Loss", loss.data[0], train_batch_out)

                    train_batch_out += 1

                    if i_batch == 0:
                        for name, param in model.named_parameters():
                            writer.add_histogram("OracleParams/Oracle_" + name, param.data, epoch, bins='auto')

                        if epoch > 0 and epoch % 5 == 0:
                            labels = list(OrderedDict(sorted({int(k): v for k, v in i2word.items()}.items())).values())
                            writer.add_embedding(model.module.word_embeddings.weight.data, metadata=labels,
                                                 tag='oracle word embedding', global_step=int(epoch / 5))

                        if epoch == 0:
                            writer.add_graph(model, pred_answer)

                elif split == 'val' and exp_config['logging']:
                    writer.add_scalar("Validation/Batch Accurarcy", accuracy[-1], valid_batch_out)
                    writer.add_scalar("Validation/Batch Loss", loss.data[0], valid_batch_out)
                    valid_batch_out += 1

            if split == 'train':
                train_accuracy = np.mean(accuracy)
            elif split == 'val':
                val_accuracy = np.mean(accuracy)

        if exp_config['save_models']:
            if not os.path.exists(exp_config['save_models_path']):
                os.makedirs(exp_config['save_models_path'])
            torch.save(
                model.state_dict(),
                os.path.join(exp_config['save_models_path'], ''.join(['oracle', args.bin_name, exp_config['ts'], str(epoch)]))
            )
            print("saving ", os.path.join(exp_config['save_models_path'], ''.join(['oracle', args.bin_name, exp_config['ts'], str(epoch)])))

        print("%s, Epoch %03d, Time taken %.2f, Training-Loss %.5f, Validation-Loss %.5f, Training Accuracy %.5f, Validation Accuracy %.5f"
            % (args.exp_name, epoch, time() - start, torch.mean(train_loss), torch.mean(val_loss), train_accuracy, val_accuracy))

        if exp_config['logging']:
            writer.add_scalar("Training/Epoch Loss", torch.mean(train_loss), epoch)
            writer.add_scalar("Training/Epoch Accuracy", train_accuracy, epoch)

            writer.add_scalar("Validation/Epoch Loss", torch.mean(val_loss), epoch)
            writer.add_scalar("Validation/Epoch Accuracy", val_accuracy, epoch)
