import argparse

parser = argparse.ArgumentParser(description='MARL')

exp_logger = parser.add_argument_group('experiment_logger')
exp_logger.add_argument('--exp_logger_log_level', type=str, default='INFO', help='logger level')
exp_logger.add_argument('--exp_logger_custom_file_handler_path', type=bool, default=False,
                        help='whether custom logger directory')
exp_logger.add_argument('--exp_logger_file_handler_path', type=str, default='', help='logger filename')

exp_running = parser.add_argument_group('experiment_running')
exp_running.add_argument('--exp_running_seed', type=int, default=0, help='set seed')
exp_running.add_argument('--exp_running_episode', type=int, default=int(1e4), help='max running episode')
exp_running.add_argument('--exp_running_timestep', type=int, default=int(1e2), help='max running timestep')
exp_running.add_argument('--exp_running_buffer_size', type=int, default=128, help='max running buffer size')
exp_running.add_argument('--exp_running_batch_size', type=int, default=32, help='max running batch size')
exp_running.add_argument('--exp_running_mode', type=str, default='exp',
                         help="mode regarding store results, ['exp', 'tune', 'debug']")

model = parser.add_argument_group('model')
model.add_argument('--model_hidden_l1_dim', type=int, default=64, help='hidden size for RNN 1st layer')
model.add_argument('--model_hidden_l2_dim', type=int, default=128, help='hidden size for RNN 2nd layer')
model.add_argument('--model_optimizer', type=str, default='rmsprop', help='[rmsprop, adam]')
model.add_argument('--model_loss', type=str, default='mse', help='criterion for training')




