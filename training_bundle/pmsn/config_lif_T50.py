"""PMSN configuration for lif at T=50."""

Reconstructed from checkpoints/pmsn_lif_T50/checkpoint.pth["config"].

CONFIG = {
    'T': 50,
    'batch_size': 40,
    'ckpt_path': '/content/work/exp/pmsn_lif_T50_fill_5ep',
    'd_input': 140,
    'd_model': 352,
    'd_state': 1,
    'data_path': '/content/shd',
    'dataset': 'SHD',
    'device': 'cuda:0',
    'dropout': 0.2,
    'epochs': 5,
    'log_interval': 300,
    'lr': 0.01,
    'neuron_lr': 0.001,
    'pretrain': None,
    'seed': 1111,
    'start_epoch': 0,
    'step_lr': False,
    'weight_decay': 0,
}
