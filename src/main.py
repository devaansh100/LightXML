import sys
import random
import numpy as np
from apex import amp
from model import LightXML

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from transformers import AdamW

import torch

from torch.utils.data import DataLoader
from dataset import MDataset, createDataCSV
from log import Logger

def load_group(dataset, group_tree=0):
    if dataset == 'wiki500k':
        return np.load(f'./data/Wiki-500K/label_group{group_tree}.npy', allow_pickle=True)
    elif dataset == 'amazon670k':
        return np.load(f'./data/Amazon-670K/label_group{group_tree}.npy', allow_pickle=True)

def train(model, optimizer_g, optimizer_m, df, label_map, max_only_p5 = 0, epoch = 0):
    tokenizer = model.get_tokenizer()

    if args.dataset in ['wiki500k', 'amazon670k']:
        group_y = load_group(args.dataset, args.group_y_group)
        train_d = MDataset(df, 'train', tokenizer, label_map, args.max_len, group_y=group_y,
                           candidates_num=args.group_y_candidate_num)#, token_type_ids=token_type_ids)
        test_d = MDataset(df, 'test', tokenizer, label_map, args.max_len, 
                           candidates_num=args.group_y_candidate_num)#, token_type_ids=token_type_ids)

        train_d.tokenizer = model.get_fast_tokenizer()
        test_d.tokenizer = model.get_fast_tokenizer()

        trainloader = DataLoader(train_d, batch_size=args.batch, num_workers=5,
                                 shuffle=True)
        testloader = DataLoader(test_d, batch_size=args.batch, num_workers=5,
                                shuffle=False)
        if args.valid:
            valid_d = MDataset(df, 'valid', tokenizer, label_map, args.max_len, group_y=group_y,
                               candidates_num=args.group_y_candidate_num)
            validloader = DataLoader(valid_d, batch_size=args.batch, num_workers=0, 
                                     shuffle=False)
    else:
        train_d = MDataset(df, 'train', tokenizer, label_map, args.max_len)
        test_d = MDataset(df, 'test', tokenizer, label_map, args.max_len)
        trainloader = DataLoader(train_d, batch_size=args.batch, num_workers=2,
                                 shuffle=True)
        testloader = DataLoader(test_d, batch_size=args.batch, num_workers=1,
                                shuffle=False)

    model.cuda()
        
    model, optimizer_m = amp.initialize(model, optimizer_m, opt_level="O1")
    model.generator, optimizer_g = amp.initialize(model.generator, optimizer_g, opt_level="O1")

    for epoch_c in range(epoch+1, args.epoch+5):
        train_loss = model.one_epoch(epoch_c, trainloader, optimizer_m, optimizer_g, mode='train',
                                     eval_loader=validloader if args.valid else testloader,
                                     eval_step=args.eval_step, log=LOG)

        if args.valid:
            ev_result = model.one_epoch(epoch_c, validloader, optimizer_m, optimizer_g, mode='eval')
        else:
            ev_result = model.one_epoch(epoch_c, testloader, optimizer_m, optimizer_g, mode='eval')

        g_p1, g_p3, g_p5, p1, p3, p5 = ev_result

        log_str = f'{epoch:>2}: {p1:.4f}, {p3:.4f}, {p5:.4f}, train_loss:{train_loss}'
        if args.dataset in ['wiki500k', 'amazon670k']:
            log_str += f' {g_p1:.4f}, {g_p3:.4f}, {g_p5:.4f}'
        if args.valid:
            log_str += ' valid'
        LOG.log(log_str)

        torch.save({
                'epoch': epoch_c,
                'model_state_dict': model.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_m_state_dict_m': optimizer_m.state_dict(),
                'max_only_p5': max_only_p5
                }, f'/content/drive/MyDrive/XMC/LightXML/models/checkpoint-{get_exp_name()}-lc.pth')
        print(f'Saving checkpoint at {epoch_c} epochs')

        if max_only_p5 < p5:
            torch.save({
                'epoch': epoch_c,
                'model_state_dict': model.state_dict(),
                'optimizer_g_state_dict_g': optimizer_g.state_dict(),
                'optimizer_m_state_dict_m': optimizer_m.state_dict(),
                'max_only_p5': max_only_p5
                }, f'/content/drive/MyDrive/XMC/LightXML/models/model-{get_exp_name()}-lc.pt')
            print(f'max_only_p5 reduced from {p5} to {max_only_p5}. Saving model at {epoch_c} epochs')
            max_only_p5 = p5

        if epoch_c >= args.epoch + 5 and max_only_p5 != p5:
            break


def get_exp_name():
    name = [args.dataset, '' if args.bert == 'bert-base' else args.bert]
    if args.dataset in ['wiki500k', 'amazon670k']:
        name.append('t'+str(args.group_y_group))

    return '_'.join([i for i in name if i != ''])


def init_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, required=False, default=16)
parser.add_argument('--update_count', type=int, required=False, default=1)
parser.add_argument('--lr', type=float, required=False, default=0.0001)
parser.add_argument('--seed', type=int, required=False, default=6088)
parser.add_argument('--epoch', type=int, required=False, default=20)
parser.add_argument('--dataset', type=str, required=False, default='eurlex4k')
parser.add_argument('--bert', type=str, required=False, default='bert-base')

parser.add_argument('--max_len', type=int, required=False, default=512)

parser.add_argument('--valid', action='store_true')

parser.add_argument('--swa', action='store_true')
parser.add_argument('--swa_warmup', type=int, required=False, default=10)
parser.add_argument('--swa_step', type=int, required=False, default=100)

parser.add_argument('--group_y_group', type=int, default=0)
parser.add_argument('--group_y_candidate_num', type=int, required=False, default=3000)
parser.add_argument('--group_y_candidate_topk', type=int, required=False, default=10)

parser.add_argument('--eval_step', type=int, required=False, default=20000)

parser.add_argument('--hidden_dim', type=int, required=False, default=300)

parser.add_argument('--eval_model', action='store_true')

parser.add_argument('--load_chk', action='store_true', required = False)
parser.add_argument('--load_chk_name', type=str, required = False, default = 'model-eurlex4k')

args = parser.parse_args()

if __name__ == '__main__':
    init_seed(args.seed)

    print(get_exp_name())

    LOG = Logger('log_'+get_exp_name())
    
    print(f'load {args.dataset} dataset...')
    df, label_map = createDataCSV(args.dataset)
    if args.valid:
        train_df, valid_df = train_test_split(df[df['dataType'] == 'train'],
                                              test_size=4000,
                                              random_state=1240)
        df.iloc[valid_df.index.values, 2] = 'valid'
        print('valid size', len(df[df['dataType'] == 'valid']))

    print(f'load {args.dataset} dataset with '
          f'{len(df[df.dataType =="train"])} train {len(df[df.dataType =="test"])} test with {len(label_map)} labels done')
    no_decay = ['bias', 'LayerNorm.weight']
    if args.dataset in ['wiki500k', 'amazon670k']:
        group_y = load_group(args.dataset, args.group_y_group)
        _group_y = []
        for idx, labels in enumerate(group_y):
            _group_y.append([])
            for label in labels:
                _group_y[-1].append(label_map[label])
            _group_y[-1] = np.array(_group_y[-1])
        group_y = np.array(_group_y)

        model = LightXML(n_labels=len(label_map), group_y=group_y, bert=args.bert,
                          update_count=args.update_count,
                          use_swa=args.swa, swa_warmup_epoch=args.swa_warmup, swa_update_step=args.swa_step,
                          candidates_topk=args.group_y_candidate_topk,
                          hidden_dim=args.hidden_dim)
    else:
        model = LightXML(n_labels=len(label_map), bert=args.bert,
                         update_count=args.update_count,
                         use_swa=args.swa, swa_warmup_epoch=args.swa_warmup, swa_update_step=args.swa_step)
    model.cuda()
    optimizer_g_grouped_parameters = [
    {'params': [p for n, p in model.generator.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.generator.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer_m_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer_g = AdamW(optimizer_g_grouped_parameters, lr=args.lr)#, eps=1e-8)
    optimizer_m = AdamW(optimizer_m_grouped_parameters, lr=args.lr)#, eps=1e-8)
    if args.load_chk:
        checkpoint = torch.load(f'/content/drive/MyDrive/XMC/LightXML/models/checkpoint-{get_exp_name()}-lc.pth', map_location = torch.device('cuda'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_m.load_state_dict(checkpoint['optimizer_m_state_dict'])
        epoch = checkpoint['epoch']
        max_only_p5 = checkpoint['max_only_p5']
        train(model, optimizer_g, optimizer_m, df, label_map, max_only_p5, epoch)
        sys.exit(0)

    if args.eval_model and args.dataset in ['wiki500k', 'amazon670k']:
        print(f'load /content/drive/MyDrive/XMC/LightXML/models/model-{get_exp_name()}-lc.pt')
        testloader = DataLoader(MDataset(df, 'test', model.get_fast_tokenizer(), label_map, args.max_len, 
                                         candidates_num=args.group_y_candidate_num),
                                batch_size=256, num_workers=0, 
                                shuffle=False)

        group_y = load_group(args.dataset, args.group_y_group)
        validloader = DataLoader(MDataset(df, 'valid', model.get_fast_tokenizer(), label_map, args.max_len, group_y=group_y,
                                          candidates_num=args.group_y_candidate_num),
                                 batch_size=256, num_workers=0, 
                            shuffle=False)
        final_model = torch.load(f'/content/drive/MyDrive/XMC/LightXML/models/model-{get_exp_name()}-lc.pt', map_location = torch.device('cuda'))
        model.load_state_dict(final_model['model_state_dict'])
        model = model.cuda()

        print(len(df[df.dataType == 'test']))
        model.one_epoch(0, validloader, None, mode='eval')

        pred_scores, pred_labels = model.one_epoch(0, testloader, None, mode='test')
        np.save(f'/content/drive/MyDrive/XMC/LightXML/results/{get_exp_name()}-labels.npy', np.array(pred_labels))
        np.save(f'/content/drive/MyDrive/XMC/LightXML/results/{get_exp_name()}-scores.npy', np.array(pred_scores))
        sys.exit(0)

    train(model, optimizer_g, optimizer_m, df, label_map)
