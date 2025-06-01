#!/usr/bin/env python3

""" Front-end script for training a Snake agent. """

import json
import sys
import os

from keras.models import Sequential, load_model
from keras.layers import *
from keras.losses import MeanSquaredError # Ensure MeanSquaredError is imported
from keras.optimizers import RMSprop # Ensure RMSprop is imported

from snakeai.agent import DeepQNetworkAgent
from snakeai.gameplay.environment import Environment
from snakeai.utils.cli import HelpOnFailArgumentParser


def parse_command_line_args(args):
    """ Parse command-line arguments and organize them into a single structured object. """

    parser = HelpOnFailArgumentParser(
        description='Snake AI training client.',
        epilog='Example: train.py --level 10x10.json --num-episodes 30000'
    )

    parser.add_argument(
        '--level',
        required=True,
        type=str,
        help='JSON file containing a level definition.',
    )
    parser.add_argument(
        '--num-episodes',
        required=True,
        type=int,
        default=30000, # Default is handled by the original script if not specified by user
        help='The number of episodes to run consecutively.',
    )
    parser.add_argument(
        '--load-model',
        type=str,
        default=None, # Changed default to None for clearer checking
        help='File containing a pre-trained agent model to load before training.',
    )
    parser.add_argument(
        '--initial-episode',
        type=int,
        default=0,
        help='The episode number to start training from. Used for resuming training.',
    )

    return parser.parse_args(args)


def create_snake_environment(level_filename):
    """ Create a new Snake environment from the config file. """

    with open(level_filename) as cfg:
        env_config = json.load(cfg)

    return Environment(config=env_config, verbose=1)


def create_dqn_model(env, num_last_frames):
    """
    Build a new DQN model to be used for training.
    
    Args:
        env: an instance of Snake environment. 
        num_last_frames: the number of last frames the agent considers as state.

    Returns:
        A compiled DQN model.
    """

    model = Sequential()

    # Convolutions.
    model.add(Conv2D(
        16,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format='channels_first',
        input_shape=(num_last_frames, ) + env.observation_shape
    ))
    model.add(Activation('relu'))
    model.add(Conv2D(
        32,
        kernel_size=(3, 3),
        strides=(1, 1),
        data_format='channels_first'
    ))
    model.add(Activation('relu'))

    # Dense layers.
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(env.num_actions))

    model.summary()
    model.compile(optimizer=RMSprop(), loss=MeanSquaredError()) # Use explicit classes

    return model


def main():
    parsed_args = parse_command_line_args(sys.argv[1:])

    env = create_snake_environment(parsed_args.level)
    model = None # Initialize model to None
    
    if parsed_args.load_model and os.path.exists(parsed_args.load_model):
        print(f'Loading pre-trained model from "{parsed_args.load_model}"')
        try:
            model = load_model(parsed_args.load_model)
        except TypeError as e:
            if 'Could not locate function' in str(e) and ('MSE' in str(e) or 'mse' in str(e)):
                print(f"Failed to load model with default MSE recognition due to: {e}. Trying with custom_objects...")
                model = load_model(parsed_args.load_model, custom_objects={'MSE': MeanSquaredError, 'mse': MeanSquaredError, 'RMSprop': RMSprop})
            else:
                raise e # Re-throw other TypeErrors

        if model: # If model was loaded successfully
            if model.layers[-1].units != env.num_actions: # Changed to use .units
                print(
                    f'Warning: The loaded model has a different number of actions ({model.layers[-1].units}) '
                    f'than the environment ({env.num_actions}). It may not work correctly.'
                )
            # Re-compile the model to ensure optimizer is correctly initialized for further training
            print("Re-compiling loaded model...")
            model.compile(optimizer=RMSprop(), loss=MeanSquaredError())
    
    if model is None: # If model wasn't loaded or loading failed and wasn't handled
        if parsed_args.load_model: # If a path was given but model is still None
             print(f'Warning: Pre-trained model at "{parsed_args.load_model}" could not be loaded or was not found. Starting training from scratch.')
        else:
             print(f'No pre-trained model specified. Starting training from scratch.')
        model = create_dqn_model(env, num_last_frames=4)
        
    agent = DeepQNetworkAgent(
        model=model,
        memory_size=-1,
        num_last_frames=model.input_shape[1]
    )
    agent.train(
        env,
        batch_size=64,
        num_episodes=parsed_args.num_episodes,
        checkpoint_freq=parsed_args.num_episodes // 10,
        discount_factor=0.95,
        start_episode=parsed_args.initial_episode
    )


if __name__ == '__main__':
    main()
