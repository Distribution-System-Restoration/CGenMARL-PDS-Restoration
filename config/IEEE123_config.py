import argparse

def get_args():
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG environment")
    parser.add_argument("-f")

    parser.add_argument("--max_train_steps", type=int, default=int(0.5*1e4), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=20, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=50, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")
    parser.add_argument("--max_action", type=float, default=1.0, help="Max action")

    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--noise_std_init", type=float, default=0.5, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0.1, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_decay_steps", type=float, default=100, help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
    parser.add_argument("--lr_a", type=float, default=1e-3, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-3, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Softly update the target network")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--N", type=int, default=3)

    args = parser.parse_args()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps
    args.N = 5

    args.obs_dim_n = [10,5,3,5,3]  # obs dimensions of N agents
    args.action_dim_n = [10,5,3,5,3]  # actions dimensions of N agents

    return args
