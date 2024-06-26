import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    
    # Basic parameters
    parser.add_argument("--dataset", type=str, default="NYC")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    
    # Data parameters
    parser.add_argument('--num_nodes', type=int, default=206)
    parser.add_argument('--input_len', type=int, default=12)
    parser.add_argument('--output_len', type=int, default=12)
    parser.add_argument('--tod_size', type=int, default=48)
    
    # Data split parameters
    parser.add_argument('--train_size', type=float, default=0.7)
    parser.add_argument('--val_size', type=float, default=0.1)
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0003)
    # parser.add_argument('--milestones', nargs='+', type=int, default=[50, 120, 200])
    # parser.add_argument('--milestones', nargs='+', type=int, default=[50, 120])
    parser.add_argument('--steps', nargs='+', type=int, default=[30, 50])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--threshold', type=float, default=0.000001)
    
    # Model arguments
    parser.add_argument('--obser_dim', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--obser_embed_dim', type=int, default=24)
    parser.add_argument('--tod_embed_dim', type=int, default=24)
    parser.add_argument('--dow_embed_dim', type=int, default=24)
    parser.add_argument('--timestamp_embed_dim', type=int, default=12)
    parser.add_argument('--spatial_embed_dim', type=int, default=12)
    parser.add_argument('--temporal_embed_dim', type=int, default=60)
    parser.add_argument('--prompt_dim', type=int, default=72)
    parser.add_argument('--self_atten_dim', type=int, default=168)
    parser.add_argument('--cross_atten_dim', type=int, default=24)
    parser.add_argument('--feed_forward_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    return parser
