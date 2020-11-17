import argparse

def ir_baseline_opts(parser):
    parser.add_argument('-train', required=True,
                        help="Path to train json file")
    parser.add_argument('-test', required=True,
                        help="Path to test json file")
    parser.add_argument('-output', required=True,
                        help="Path to output result file")
    parser.add_argument('-model', default='tfidf', type=str,
                        help="IR model option: bm25")

def preprocess_opts(parser):
    """
        Options for data pre-processing
        """
    parser.add_argument('-lm', default='bert', type=str,
                        help="Language model option: bert")
    parser.add_argument('-input',  required=True,
                        help="Path to input json file")
    parser.add_argument('-output', required=True,
                        help="Path to output file")

def model_opts(parser):
    """
    Options for model construction
    """
    parser.add_argument('-hidden_size', type=int, default=256,
                        help='Size of hidden states')
    parser.add_argument('-method', default='opt', type=str,
                        help="Similarity aggregation method ap[Assignment Problem], opt[Optimal Transport], max_p[Avg_Max_P], max_s[Avg_Max_S], att_[Attention]")


def train_opts(parser):
    """
    Options for training
    """
    parser.add_argument('-epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('-learning_rate', type=float, default=0.01,
                        help='Number of epochs')
    parser.add_argument('-batch_size', type=int, default=128,
                        help='Maximum train/eval batch size')


    parser.add_argument('-alpha', type=float, default=0.5,
                        help='Margin parameter')
    parser.add_argument('-gamma', type=float, default=2.0,
                        help='Gamma')

    parser.add_argument('--seed', default=100, type=int, help='random seed')

    parser.add_argument('-cuda', action='store_true', default=True,
                        help='Use CUDA device')

    parser.add_argument('-train_dataset', default='./dataset/persona_linking_train.bert', help='Train Dataset.')
    parser.add_argument('-dev_dataset', default='./dataset/persona_linking_dev.bert', help='Dev Dataset.')
    parser.add_argument('-save_model', default='./model/',
                        help="""Model filename (the model will be saved as
                            <save_model>[method]_[epochs]_[alpha]_[learning_rate].pt""")
    parser.add_argument('-save_every', type=int, default=10,
                        help='Save models at this interval')

def test_opts(parser):
    """
    Options for test
    """
    parser.add_argument('-method', default='sparse', type=str,
                        help="rand,bert")
    parser.add_argument('-model', default=None,
                        help="""Path to the model .pt files""")
    parser.add_argument('-test_dataset',
                        help="Path to test source")
    parser.add_argument('-test_result',
                        help="Path to test result")
    parser.add_argument('-cuda', action='store_true', default=True,
                        help='Use CUDA device')
    parser.add_argument('--seed', default=100, type=int, help='random seed')
