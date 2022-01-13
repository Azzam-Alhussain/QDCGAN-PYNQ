import os 
import argparse
import torch
from trainer import DCGAN_trainer


parser = argparse.ArgumentParser(description="Quantized DCGAN Training")


def none_or_str(value):
    if value == "None":
        return None 
    return value

def none_or_int(value):
    if value == "None":
        return None
    return int(value)

parser.add_argument("--dataset", default="MNIST")
parser.add_argument("--experiments", default="./experiments")
parser.add_argument("--dry_run", action="store_true")
parser.add_argument("--log_freq", type=int, default=50)

parser.add_argument("--export", action="store_true")
parser.add_argument("--resume", type=none_or_str)
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--fid", action="store_true")

parser.add_argument("--gpus", type=none_or_str)
parser.add_argument("--num_workers", default=4, type=int)

parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--lr", default=0.0002, type=float)
parser.add_argument("--scheduler", default="FIXED", type=none_or_str)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--random_seed", default="None", type=none_or_int)
parser.add_argument("--examples", default=1, type=int)

parser.add_argument("--size", default=8, type=int)
parser.add_argument("--gen_ch", default=8, type=int)
parser.add_argument("--wbw", dest="weight_bitwidth", type=none_or_int, default=32)
parser.add_argument("--abw", dest="activation_bitwidth", type=none_or_int, default=32)
parser.add_argument("--iobw", dest="io_bitwidth", type=none_or_int, default=8)


torch.set_printoptions(precision=10)



class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such Attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such Attribute: " + name)


if __name__=="__main__":
    args = parser.parse_args()
    # creating directories
    path_args = ["experiments", "resume"]
    for path_arg in path_args:
        path = getattr(args, path_arg)
        if path is not None and not os.path.isabs(path):
            abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
            setattr(args, path_arg, abs_path)
    os.makedirs(args.experiments, exist_ok=True)
    
    if args.export:
        args.gpus = None

    # converting all arguments to dictionary
    config = objdict(args.__dict__)

    if args.evaluate:
        args.dry_run = True 


    trainer = DCGAN_trainer(config)


    if args.export:
        trainer.export_model()
        exit(0)

    if args.evaluate:
        with torch.no_grad():
            trainer.eval_model(path="test_image/out.png", examples=config.examples)
    elif args.fid:
        trainer.calculate_fid()
    else:
        trainer.train_model()





