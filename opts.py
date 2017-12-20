import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--input_json', type=str, default='data/cap_coco.json',
                    help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_dic', type=str, default='data/dic_coco.json',
                    help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--image_path', type=str, default='/home/jiasen/data/coco/images',
                    help='path to the h5file containing the image data') 
  
    parser.add_argument('--num_workers', dest='num_workers',
                    help='number of worker to load data',
                    default=10, type=int)  
    parser.add_argument('--cuda', action='store_true',
                    help='whether use cuda')
    parser.add_argument('--mGPUs', dest='mGPUs',
                    help='whether use multiple GPUs',
                    action='store_true')
    parser.add_argument('--path_opt', dest='path_opt',
                    default='cfgs/res101.yml', type=str)
    parser.add_argument('--cached_tokens', type=str, default='dataset/coco-train-idxs',
                    help='Cached token file for calculating cider score during self critical training.')

    # Model settings
    parser.add_argument('--rnn_size', type=int, default=512,
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                    help='rnn, gru, or lstm')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                    help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--att_hid_size', type=int, default=512,
                    help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                    help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                    help='2048 for resnet, 512 for vgg')
    parser.add_argument('--image_size', type=int, default=256,
                    help='image random crop size')
    parser.add_argument('--image_crop_size', type=int, default=224,
                    help='image random crop size')
    parser.add_argument('--att_size', type=int, default=14,
                    help='conv feat size after spatial adaptive pooling.')    

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=30,
                    help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=50,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--self_critical', type=bool, default=False,
                    help='whether use self critical training.')
    parser.add_argument('--seq_per_img', type=int, default=5,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
    parser.add_argument('--seq_length', type=int, default=16, help='')
    parser.add_argument('--beam_size', type=int, default=1,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    # Schedule Sampling.
    parser.add_argument('--scheduled_sampling_start', type=int, default=-1, 
                    help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5, 
                    help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05, 
                    help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25, 
                    help='Maximum scheduled sampling prob.')

    #Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                    help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1, 
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay')

    # Optimization: for the CNN
    parser.add_argument('--finetune_cnn', action='store_true',
                    help='finetune CNN')  
    parser.add_argument('--fixed_block', type=float, default=1,
                    help='fixed cnn block when training. [0-4] \
                            0:finetune all block, 4: fix all block')
    parser.add_argument('--cnn_optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--cnn_optim_alpha', type=float, default=0.9,
                    help='cnn alpha for adam')
    parser.add_argument('--cnn_optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--cnn_learning_rate', type=float, default=4e-4,
                    help='cnn learning rate')
    parser.add_argument('--cnn_weight_decay', type=float, default=0,
                    help='weight_decay')

    # set training session
    parser.add_argument('--start_from', type=str, default=None,
                    help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                    """)
    parser.add_argument('--id', type=str, default='',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')


    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=3200,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--checkpoint_path', type=str, default='save',
                    help='directory to store checkpointed models')
    parser.add_argument('--language_eval', type=int, default=1,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')       
    parser.add_argument('--disp_interval', type=int, default=100,
                    help='how many iteration to display an loss.')       
    parser.add_argument('--losses_log_every', type=int, default=10,
                    help='how many iteration for log.')   
    args = parser.parse_args()

    return args