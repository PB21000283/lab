import argparse
import os


parser = argparse.ArgumentParser()

# ===================== gpu id ===================== #
parser.add_argument('--gpu',            type=int,   default=0)

# =================== random seed ================== #
parser.add_argument('--seed',           type=int,   default=1234)

# ==================== dataset ===================== #
parser.add_argument('--train_file',
                    default='data/bj_train_set.npz')
parser.add_argument('--val_file',
                    default='data/bj_val_set.npz')
parser.add_argument('--test_file',
                    default='data/bj_test_set.npz')
parser.add_argument('--poi_file',
                    default='data/bj_transh_poi_10.npz')
parser.add_argument('--nodes',          type=int,   default=28342,
                    help='Newyork=95581, Beijing=28342')

# ===================== model ====================== #
parser.add_argument('--latent_dim',     type=int,   default=128)
parser.add_argument('--num_heads',      type=int,   default=8)
parser.add_argument('--n_epochs',       type=int,   default=300)
parser.add_argument('--batch_size',     type=int,   default=256)
parser.add_argument('--lr',             type=float, default=5e-4,
                    help='5e-4 for Beijing, 1e-3 for Newyork')
parser.add_argument('--save_epoch_int', type=int,   default=1)
parser.add_argument('--save_folder',                default='saved_models')

##TODO,新增args

    

        self.future_mask=args.future_mask

        self.d_model = args.d_model

        self.n_layers =args.n_layers

        self.attn_heads = args.attn_heads

        self.mlp_ratio =args.mlp_ratio

        self.dropout = args.dropout

        self.drop_path = args.drop_path

        self.lape_dim = args.lape_dim

        self.attn_drop = args.attn_drop

        self.type_ln = args.type_ln

        self.vocab_size = args.vocab_size##TODO,args

        self.pooling = args.pooling##池化方式


        self.add_cls = args.add_cls
        
dataset = DataLoader(data_, batch_size=batchsize, shuffle=True, ##TODO
                        collate_fn=lambda x: collate_fn_neg(x, max_len=args.seq_len,
                                                            vocab=args.vocab, add_cls=args.add_cls))
{
  "n_layers": 6,
  "d_model": 256,
  "attn_heads": 8,
  "max_epoch": 30,
  "batch_size": 64,
  "grad_accmu_steps": 1,
  "learning_rate": 2e-4,
  "dataset": "bj",
  "roadnetwork": "bj_roadmap_edge_bj_True_1_merge",
  "geo_file": "bj_roadmap_edge_bj_True_1_merge_withdegree",
  "rel_file": "bj_roadmap_edge_bj_True_1_merge_withdegree",
  "merge": true,
  "min_freq": 1,
  "seq_len": 128,
  "test_every": 50,
  "temperature": 0.05,
  "contra_loss_type": "simclr",
  "classify_label": "vflag",
  "type_ln": "post",
  "add_cls": true,
  "add_time_in_day": true,
  "add_day_in_week": true,
  "add_pe": true,
  "add_temporal_bias": true,
  "temporal_bias_dim": 64,
  "use_mins_interval": false,
  "add_gat": true,
  "gat_heads_per_layer": [8, 16, 1],
  "gat_features_per_layer": [16, 16, 256],
  "gat_dropout": 0.1,
  "gat_K": 1,
  "gat_avg_last": true,
  "load_trans_prob": true,
  "append_degree2gcn": true,
  "normal_feature": false,
  "pooling": "cls"
}

parser.add_argument('--batch_size',     type=int,   default=256)
parser.add_argument('--batch_size',     type=int,   default=256)
parser.add_argument('--batch_size',     type=int,   default=256)
parser.add_argument('--batch_size',     type=int,   default=256)
parser.add_argument('--batch_size',     type=int,   default=256)
parser.add_argument('--batch_size',     type=int,   default=256)
parser.add_argument('--batch_size',     type=int,   default=256)
parser.add_argument('--batch_size',     type=int,   default=256)
parser.add_argument('--batch_size',     type=int,   default=256)
parser.add_argument('--batch_size',     type=int,   default=256)
parser.add_argument('--batch_size',     type=int,   default=256)
parser.add_argument('--batch_size',     type=int,   default=256)
parser.add_argument('--batch_size',     type=int,   default=256)
parser.add_argument('--batch_size',     type=int,   default=256)
##TODO



args = parser.parse_args()

# setup device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
