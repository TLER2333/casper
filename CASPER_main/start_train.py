import argparse
import os
import sys
import torch
import torch.distributed as dist

sys.path.append(os.getcwd())

from casper.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="Path to the config file")
    parser.add_argument("--seed", type=int, default=None, help="Override the seed in the config")
    parser.add_argument("--local-rank", type=int, default=os.getenv('LOCAL_RANK', -1), help="Local rank")
    args = parser.parse_args()

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        local_rank = args.local_rank if args.local_rank != -1 else 0
        world_rank = 0
        world_size = 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method='env://')
        if world_rank == 0:
            print(f"[Init] Distributed training initialized. World size: {world_size}")
    else:
        print("[Init] Running in single process mode.")

    try:
        with Trainer(
            config_path=args.config,
            local_rank=local_rank,
            world_rank=world_rank,
            seed=args.seed,
        ) as trainer:
            trainer.train()
    except Exception as e:
        print(f"[Error] Rank {world_rank} crashed with error: {e}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
