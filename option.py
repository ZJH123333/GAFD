import argparse

def parse():
    parser = argparse.ArgumentParser(description='KD')

    # train
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed')

    parser.add_argument('--k_fold', type=int, default=5,
                        help='the fold number')

    parser.add_argument('--minibatch_size', type=int, default=4, # 4
                        help='batch size')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')

    parser.add_argument('--weight_decay', type=float, default=1e-5, # 1e-5
                        help='weight_decay')

    parser.add_argument('--num_epochs', type=int, default=5, # 5
                        help='local epochs')

    parser.add_argument('--num_iters', type=int, default=50, # 50
                        help='communication rounds')

    # adj and feature
    parser.add_argument('--sparsity', type=int, default=30, # 30
                        help='degree of sparsity')

    parser.add_argument('--self_loop', type=bool, default=True,
                        help='whether to include self-loops when computing the sparsity threshold')

    parser.add_argument('--add_edge_weight', type=bool, default=False, # False
                        help='if add edge weight for adj')

    parser.add_argument('--n2p', type=bool, default=True,
                        help='whether to convert negative edge weights into positive edge weights')

    parser.add_argument('--fisher_z', type=bool, default=True,
                        help='whether to apply fisher_z transformation to PC')

    # model
    parser.add_argument('--input_dim', type=int, default=116,
                        help='input dimension of extractor')

    parser.add_argument('--hidden_dim', type=int, default=128, # 128
                        help='hidden dimension of extractor')

    parser.add_argument('--output_dim', type=int, default=32, # 32
                        help='output dimension of extractor')

    parser.add_argument('--d_k', type=int, default=64,  # 64
                        help='d_k in transformer layer')

    parser.add_argument('--d_v', type=int, default=64,  # 64
                        help='d_v in transformer layer')

    parser.add_argument('--d_ff', type=int, default=112,  # 112
                        help='d_ff in transformer layer')

    parser.add_argument('--num_layers', type=int, default=2, # 2
                        help='number of transformer layers')

    parser.add_argument('--num_heads', type=int, default=2, # 2
                        help='number of transformer heads')

    parser.add_argument('--fc_dim', type=int, default=6670,  # 6670
                        help='global fc dimension')

    parser.add_argument('--fc_hidden_dim', type=int, default=32, # 32
                        help='global fc hidden dimension')

    parser.add_argument('--num_class', type=int, default=2,
                        help='number of task categories')

    # aug
    parser.add_argument('--pe', type=float, default=0.3, # 0.3
                        help='edge drop rate')

    parser.add_argument('--pf', type=float, default=0.2, # 0.2
                        help='node feature drop rate')

    parser.add_argument('--p_threshold', type=float, default=0.7, # 0.7
                        help='drop rate threshold')

    # kd
    parser.add_argument('--temperature', type=float, default=1, # 1
                        help='temperature')

    parser.add_argument('--alpha', type=float, default=0.7, # 0.7
                        help='alpha')

    # server
    parser.add_argument('--avg_weighted', type=bool, default=True,
                        help='if use weighted average to aggregate scores on the server side')

    parser.add_argument('--max_weighted', type=bool, default=False,
                        help='if use weighted max to aggregate scores on the server side')


    # evaluation
    parser.add_argument('--acc_only', type=bool, default=False,
                        help='if calculate accuracy only')

    # save
    parser.add_argument('--save_root_path', type=str, default='./results',
                        help='root directory of results')

    argv = parser.parse_args()
    return argv