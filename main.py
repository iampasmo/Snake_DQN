#%%
import argparse

from snake_RL import play_ai, play_human


#%%


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Snake Game with DQN')
    parser.add_argument('player', metavar='player', type=str,
                        help='player : human / ai')
    
    parser.add_argument('--load_model', type=bool, default=True, required=False,
                        help='If you want to train with your own, set this value False ')    
    parser.add_argument('--load_model_epochs', type=int, default=7000, required=False,
                        help='load_model_epochs')

    
    args = parser.parse_args()
    
    if args.player == "human":
        play_human()
    else :
        play_ai(args.load_model_epochs, args.load_model)
    