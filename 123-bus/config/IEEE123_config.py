import argparse


def get_args():
    parser = argparse.ArgumentParser("Hyperparameters Setting for CGAN-SAC environment")
    parser.add_argument("-f")
    parser.add_argument("--max_train_steps", type=int, default=int(15e3), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=15, help="Maximum number of steps per episode")
    parser.add_argument("--buffer_size", type=int, default=int(2e6), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=128, help="The number of neurons in hidden layers of the neural networks")
    parser.add_argument("--epsilon_init", type=float, default=1.0, help="initialization epsilon of epsilon-greedy")
    parser.add_argument("--epsilon_min", type=float, default=0.1, help="minimum epsilon epsilon of epsilon-greedy")
    parser.add_argument("--decay_rate", type=float, default=0.0001, help="epsilon decay in each step")
    parser.add_argument("--epsilon_pre_steps", type=int, default=1e3, help="steps before decay start")
    parser.add_argument("--M", type=float, default=1, help="negative reward coefficient")
    parser.add_argument("--lr_a", type=float, default=1e-3, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-3, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Softly update the target network")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip") 
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--P_gen", type=int, default=int(2400), help="generated power")
    parser.add_argument("--average_window_size", type=int, default=10)
    
    
    args = parser.parse_args()
    args.N = 5
    args.obs_dim_n = [10,5,3,5,3]  # obs dimensions of N agents
    args.action_dim_n = [10,5,3,5,3]  # actions dimensions of N agents
    
    return args