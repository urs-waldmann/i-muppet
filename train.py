import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import sys
import os
import argparse
sys.path.insert(0, os.path.abspath('./utils'))
from engine import train_one_epoch
import utils as utils
import misc_utils
import dataset_utils
import network_utils


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        type=str,
                        required=True,
                        # default='./experiments/muppet_600.yaml',
                        help="Input: Specify the path to the configuration file.")
    parser.add_argument("--display_data",
                        action='store_true',
                        help="Visualization: Display images and targets of data.")

    args = parser.parse_args()

    return args


def main(args):

    ################
    ### Training ###
    ################

    print('[INFO] prepare training...')

    # load config
    config = misc_utils.load_config(args.config)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #################
    ### Variables ###
    species = config.species  # ['pigeon', 'cowbird', 'mouse']

    if species == 'pigeon':
        ## Pigeons
        session_train = config.dataset.training_session
        # Set pretrained_model to None if needed
        pretrained_model = config.dataset.pretrained_model
        flip_probability = config.data_augmentation.flip_probability
        scale_percentages_range = config.data_augmentation.scale_percentages_range
    elif species == 'cowbird':
        ## Cowbirds
        session_train, pretrained_model, flip_probability, scale_percentages_range = None, None, None, None
    elif species == 'mouse':
        ## Mice
        session_train = config.dataset.training_session
        # Set pretrained_model to None if needed
        pretrained_model = config.dataset.pretrained_model
        flip_probability = config.data_augmentation.flip_probability
        scale_percentages_range = config.data_augmentation.scale_percentages_range
    else:
        assert False

    num_workers = config.num_workers

    brightness = config.data_augmentation.brightness
    contrast = config.data_augmentation.contrast
    saturation = config.data_augmentation.saturation
    hue = config.data_augmentation.hue
    # float: 0 gives blurred image, 1 gives original image, 2 increases sharpness by factor 2
    sharpness_factor = config.data_augmentation.sharpness_factor
    sharpness_prob = config.data_augmentation.sharpness_prob  # probability
    batch_size_train = config.hyperparameters.batch_size_train  # Batch size for training
    learning_rate = config.hyperparameters.learning_rate  # Learning rate of optimizer
    momentum = config.hyperparameters.momentum  # Momentum of optimizer
    weight_decay = config.hyperparameters.weight_decay  # Weight decay of optimizer (L2 penalty).
    step_size = config.hyperparameters.step_size  # Period of learning rate decay
    gamma = config.hyperparameters.gamma  # Multiplicative factor of learning rate decay. Default: 0.1
    num_epochs = config.hyperparameters.num_epochs  # Number of epochs
    print_freq = config.print_freq  # Iterations printed
    save_model_at = './data/weights/my_weights'
    #################

    # create 'my_weights' folder
    misc_utils.create_folder(save_model_at)
    # set where to store trained weights
    save_model_at = os.path.join(save_model_at, args.config[args.config.rfind('/') + 1:args.config.rfind('.')] + '.pt')

    print('\nSpecies:', config.species)

    if (species == 'pigeon') or (species == 'mouse'):
        print('Training session:', session_train)
        if pretrained_model is not None:
            # set path of pre-trained model
            pretrained_model = os.path.join('./data/weights', pretrained_model + '.pt')
            print('Pre-trained model:', pretrained_model)
        else:
            pass
    else:
        pass
    print('New model will be stored at', save_model_at, '\n')

    ### Load dataset ###
    print('[INFO] load data...')

    if args.display_data:
        transform = transforms.Compose(  # to display images
            [transforms.ToTensor(),
             transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
             transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor, p=sharpness_prob)
             ])
    else:
        transform = transforms.Compose(  # for training
            [transforms.ToTensor(),
             transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
             transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor, p=sharpness_prob),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
             ])

    trainset = dataset_utils.load_dataset(species=species,
                                          session=session_train,
                                          flip_probability=flip_probability,
                                          scale_percentages_range=scale_percentages_range,
                                          transform=transform)

    trainloader = DataLoader(trainset,
                             batch_size=batch_size_train,
                             shuffle=True,
                             num_workers=num_workers,
                             collate_fn=utils.collate_fn)

    if args.display_data:
        # Display image and targets.
        misc_utils.display_dataset(trainloader, species=species, check_left_right=False)
    else:
        ### Load Keypoint R-CNN model ###
        net = network_utils.load_network(network_name='KeypointRCNN',
                                         looking_for_object=species,
                                         eval_mode=False,
                                         pre_trained_model=pretrained_model,
                                         device=device
                                         )

        ### Define optimizer and learning rate scheduler ###
        print('[INFO] define optimizer and learning rate scheduler...')

        # construct an optimizer
        params = [p for p in net.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        ### Training ###
        print('[INFO] start training...\n')
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            # train for one epoch, printing every 'print_freq' iterations
            train_one_epoch(net, optimizer, trainloader, device, epoch, print_freq=print_freq)
            # update the learning rate
            lr_scheduler.step()

        ### Save model ###
        # Save state_dict only
        print('\n[INFO] save our trained model...')
        torch.save(net.state_dict(), save_model_at)

    print('[INFO] finished')


if __name__ == '__main__':

    my_args = parse_args()
    print("args: {}".format(my_args))

    main(my_args)
