import re
import time
import queue
import threading
import json
import torch
import torch.distributed as dist
from types import SimpleNamespace
from pathlib import Path
from collections import defaultdict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from .logger import Logger

from casper.data_provider.data_loader import TrainLoader, ValidationLoader
from casper.model.casper import Casper as model_fn

class Trainer:
    def __init__(self, config_path, local_rank, world_rank):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        self.args = SimpleNamespace(**config_dict)
        self.args.local_rank = local_rank
        self.args.world_rank = world_rank

        self.local_rank = local_rank
        self.world_rank = world_rank
        self.device = torch.device(f"cuda:{local_rank}")
        
        self.log_dir = Path(self.args.log_dir)
        self.model_dir = Path(self.args.model_dir)
        
        if self.world_rank == 0:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.logger = Logger(self.log_dir / "train_log.txt")

        if dist.is_initialized():
            self.n_gpu = dist.get_world_size()
        else:
            self.n_gpu = 1

        self.n_accumulate = self.args.global_batch_size // self.n_gpu // self.args.local_batch_size
        
        self.train_loader = TrainLoader(self.args)
        self.val_loader = ValidationLoader(self.args)
        
        total_samples = len(self.train_loader.dataset)
        self.steps_per_epoch = max(1, total_samples // self.args.global_batch_size)
        
        self.model = model_fn(self.args).to(self.device)
        
        if self.world_rank == 0:
            self.logger.info(str(self.model))
            self.logger.info(f"Config loaded from {config_path}")
            
        if self.n_gpu > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)
        
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.args.lr, 
            betas=(0.9, 0.999), 
            eps=1e-6, 
            weight_decay=0.01
        )
        
        self.step = 0
        self.ckpt_queue = queue.Queue()
        self.load_checkpoint()
        
        if self.world_rank == 0:
            threading.Thread(target=self.sync_checkpoint, daemon=True).start()

    def adjust_learning_rate(self):
        def inverse_sqrt_root_schedule(step, n_warmup, lr):
            factor = lr * n_warmup**0.5
            return factor * min(step**-0.5, step * n_warmup**-1.5)

        lr = inverse_sqrt_root_schedule(self.step + 1, self.args.warmup_step, self.args.lr)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def train(self):
        train_iter = iter(self.train_loader)
        start_time = time.time()
        cur_time = start_time
        last_saved_time = start_time
        n_trained = 0
        scaler = GradScaler()
        
        for i in range(self.step + 1, self.args.nr_step + 1):
            self.adjust_learning_rate()
            b_losses = defaultdict(list)
            b_metrics = defaultdict(list)
            current_epoch = (i - 1) // self.steps_per_epoch

            def step_forward():
                nonlocal train_iter
                try:
                    batch_data = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch_data = next(train_iter)

                data = {k: v.to(self.device) for k, v in batch_data.items()}
                
                with autocast(enabled=self.args.mixed_precision):
                    output = self.model(data, current_epoch)
                    
                    losses = {k: v for k, v in output.items() if k.endswith("loss")}
                    metrics = {
                        k: v for k, v in output.items()
                        if k.endswith("_acc") or k.startswith("token") or k.endswith("f1") or k.endswith("_macro")
                    }
                    
                    loss = output["update_loss"] / self.n_accumulate
                    scaler.scale(loss).backward()
                    
                for k, v in losses.items():
                    b_losses[k].append(v.item())
                for k, v in metrics.items():
                    b_metrics[k].append(v)

            if self.n_gpu > 1:
                with self.model.no_sync():
                    for _ in range(self.n_accumulate - 1):
                        step_forward()
            else:
                for _ in range(self.n_accumulate - 1):
                    step_forward()

            step_forward()
            
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            
            n_trained += 1
            self.step += 1
            
            if i % 10 == 0:
                loss_str = ", ".join([f"{k}: {sum(v) / len(v):.6f}" for k, v in b_losses.items()])
                metric_str = ", ".join([f"{k}: {sum(v) / len(v):.6f}" for k, v in b_metrics.items()])
                
                now = time.time()
                speed = 1.0 / (now - cur_time)
                passed_time = (now - start_time) / 3600
                estimate_time = (self.args.nr_step - self.step) / n_trained * passed_time if n_trained > 0 else 0
                
                log_str = (
                    f"Epoch: {current_epoch}, Step: [{self.step}/{self.args.nr_step}], "
                    f"{loss_str}, {metric_str}, "
                    f"Speed: {speed:.3f} s/step, "
                    f"Passed: {passed_time:.3f} h, "
                    f"Estimate: {estimate_time:.3f} h"
                )
                
                print(log_str)
                if self.world_rank == 0:
                    self.logger.info(log_str)
                
                cur_time = now

            if (i % self.args.save_interval_step == 0 or 
                time.time() - last_saved_time > self.args.save_interval_time):
                self.save_checkpoint()
                last_saved_time = time.time()

    def save_checkpoint(self):
        if self.world_rank != 0:
            return
            
        state = {
            "step": self.step,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "args": vars(self.args)
        }
        filename = self.model_dir / f"checkpoint-step-{self.step}.pth"
        
        self.logger.info(f"Saving checkpoint: {filename} ...")
        torch.save(state, filename)
        self.ckpt_queue.put(filename)

    def load_checkpoint(self):
        latest_step = -1
        for path in self.model_dir.iterdir():
            if path.stem.startswith("checkpoint-step-"):
                try:
                    step = int(re.findall(r"\d+", path.stem)[0])
                    latest_step = max(latest_step, step)
                except IndexError:
                    pass
                    
        if latest_step != -1:
            filename = self.model_dir / f"checkpoint-step-{latest_step}.pth"
            if self.world_rank == 0:
                self.logger.info(f"Loading checkpoint: {filename} ...")
            
            checkpoint = torch.load(filename, map_location="cpu")
            self.step = checkpoint["step"]
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            
            if self.world_rank == 0:
                self.logger.info(f"Checkpoint '{filename}' (step {self.step}) loaded")

    def sync_checkpoint(self):
        while True:
            path = self.ckpt_queue.get()
            try:
                cur_step = int(re.findall(r"\d+", path.stem)[0])
                if cur_step % self.args.save_interval_step == 0:
                    for chk in self.model_dir.iterdir():
                        if chk.stem.startswith("checkpoint"):
                            try:
                                chk_step = int(re.findall(r"\d+", chk.stem)[0])
                                if chk_step % self.args.save_interval_step != 0 and chk_step < cur_step:
                                    chk.unlink()
                            except Exception:
                                pass
            except Exception:
                pass
            self.ckpt_queue.task_done()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.world_rank == 0:
            if hasattr(self, "logger") and hasattr(self.logger, "flush"):
                self.logger.flush()
            if hasattr(self, "ckpt_queue"):
                self.ckpt_queue.join()