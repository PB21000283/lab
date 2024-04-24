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

parser.add_argument('--batch_size',     type=int,   default=8)

parser.add_argument('--lr',             type=float, default=5e-4,
                    help='5e-4 for Beijing, 1e-3 for Newyork')

parser.add_argument('--save_epoch_int', type=int,   default=1)

parser.add_argument('--save_folder',                default='saved_models')

##TODO,新增args

parser.add_argument('--n_layers',     type=int,   default=6)

parser.add_argument('--d_model',     type=int,   default= 128)##TODO!!,原来是256

parser.add_argument('--attn_heads',     type=int,   default=8)

parser.add_argument('--max_epoch',     type=int,   default=30)##transformer层数

parser.add_argument('--grad_accmu_steps',     type=int,   default=1)

parser.add_argument('--add_cls',     type=bool,   default=True)

parser.add_argument('--vocab_size',          type=int,   default=28342,
                    help='Newyork=95581, Beijing=28342')

parser.add_argument('--pooling',      default='cls')

parser.add_argument('--seq_len',     type=int,   default=128)##TODO!!!,先只取一部分长度，截断，应该为682

parser.add_argument('--dropout',     type=float,   default=0.1)

parser.add_argument('--mlp_ratio',     type=int,   default=4)

parser.add_argument('--type_ln',       default='post')

parser.add_argument('--future_mask',     type=bool,   default=False)

parser.add_argument('--attn_drop',     type=float,   default=0.1)

parser.add_argument('--drop_path',     type=float,   default=0.3)

parser.add_argument('--lape_dim',     type=int,   default=256)

##TODO



args = parser.parse_args()

# setup device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
