import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # train
    ## files
    parser.add_argument('--train', default='./data/snli_train.tsv',
                             help="training data")
    parser.add_argument('--eval', default='./data/snli_test.tsv',
                             help="evaluation data")
    parser.add_argument('--train_prepro', default='./data/train_prepro',
                        help="processed training data")
    parser.add_argument('--dev_prepro', default='./data/dev_prepro',
                        help="processed dev data")

    parser.add_argument('--model_path', default='SSLNLI_E%02dL%.3f')
    parser.add_argument('--modeldir', default='./model')
    parser.add_argument('--modeldir_cls', default='./model_cls')
    parser.add_argument('--model_path_cls', default='SSLNLI_cls_E%02dL%.3f')
    parser.add_argument('--init_checkpoint', default='./model/SSLNLI_E04L2.844-40')
    parser.add_argument('--vec_path', default='./data/vec/snli_trimmed_vec.npy')

    ## vocabulary
    parser.add_argument('--vocab', default='./data/snli.vocab',
                        help="vocabulary file path")


    # training scheme
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)
    parser.add_argument('--preembedding', default=False, type=bool) #本地测试使用
    #parser.add_argument('--preembedding', default=True, type=bool) #实际训练使用
    #parser.add_argument('--early_stop', default=20, type=int)

    parser.add_argument('--lr', default=0.0003, type=float, help="learning rate")
    parser.add_argument('--local_layer', default=2, type=float)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--rand_seed', default=123, type=int)

    # model
    parser.add_argument('--d_model', default=300, type=int,
                        help="hidden dimension of interativate")
    parser.add_argument('--d_ff', default=512, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--hidden_size', default=300, type=int,
                        help="hidden_size")
    parser.add_argument('--lstm_dim', default=128, type=int,
                        help="hidden_size")

    parser.add_argument('--inter_attention', default=False, type=bool,
                        help="inter_attention")
    parser.add_argument('--num_blocks_inter', default=3, type=int,
                        help="num_blocks_inter")
    parser.add_argument('--num_dense_blocks', default=3, type=int,
                        help="num_blocks_inter")
    parser.add_argument('--num_blocks_encoder', default=4, type=int,
                        help="num_blocks_encoder")
    parser.add_argument('--num_heads', default=6, type=int,
                        help="number of attention heads")
    parser.add_argument('--maxlen', default=50, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--num_class', default=3, type=int,
                        help="number of class")
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--is_training', default=True, type=bool)

    # test
    parser.add_argument('--test_file', default='./data/snli_test.tsv')
