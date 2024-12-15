"""
MIT License

Copyright (c) 2024 Mohamed El Banani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from torch.nn.functional import interpolate
from evals.datasets.builder import build_loader
from evals.utils.losses import angular_loss
from evals.utils.metrics import evaluate_surface_norm
from evals.utils.optim import cosine_decay_linear_warmup, get_scheduler


def ddp_setup(rank: int, world_size: int, port: int = 12355):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def train(
    model,
    probe,
    train_loader,
    optimizer,
    scheduler,
    n_epochs,
    detach_model,
    rank=0,
    world_size=1,
    valid_loader=None,
    writer=None,
):
    for ep in range(n_epochs):

        if world_size > 1:
            train_loader.sampler.set_epoch(ep)

        train_loss = 0
        pbar = tqdm(train_loader) if rank == 0 else train_loader
        for i, batch in enumerate(pbar):
            images = batch["image"].to(rank)
            mask = batch["depth"].to(rank) > 0
            target = batch["snorm"].to(rank)

            optimizer.zero_grad()
            if detach_model:
                with torch.no_grad():
                    feats = model(images)
                    if type(feats) is tuple or type(feats) is list:
                        feats = [_f.detach() for _f in feats]
                    else:
                        feats = feats.detach()

            else:
                feats = model(images)

            pred = probe(feats)
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bicubic")

            if i % 100 ==0:
                writer.add_image('LOSS/pred', pred[0] , ep * len(train_loader) + i)
                writer.add_image('LOSS/gt', target[0] , ep * len(train_loader) + i)

            uncertainty = pred.shape[1] > 3
            loss = angular_loss(pred, target, mask, uncertainty_aware=uncertainty)
            loss.backward()
            optimizer.step()
            scheduler.step()

            pr_lr = optimizer.param_groups[0]["lr"]
            loss = loss.item()
            train_loss += loss

            if rank == 0:
                _loss = train_loss / (i + 1)
                pbar.set_description(f"{ep} | loss: {_loss:.4f} probe_lr: {pr_lr:.2e}")
                if i % 100 == 0:
                    writer.add_scalar(f'Accumulated Loss', _loss, ep * len(train_loader) + i)
                    writer.add_scalar(f'Loss', loss, ep * len(train_loader) + i)
                    writer.add_scalar(f'Probe LR', pr_lr, ep * len(train_loader) + i)

        train_loss /= len(train_loader)

        if rank == 0 and valid_loader is not None:
            print('Evaluate:', len(valid_loader))
            valid_loss, valid_metrics = validate(model, probe, valid_loader, writer=writer, epoch=ep)
            logger.info(f"Final valid loss       | {valid_loss:.4f}")
            writer.add_scalar(f'Eval/loss', valid_loss, (ep + 1) * len(train_loader))
            for metric in valid_metrics:
                logger.info(f"Final valid {metric:10s} | {valid_metrics[metric]:.4f}")
                writer.add_scalar(f'Eval/{metric}', valid_metrics[metric], (ep + 1) * len(train_loader))


def validate(model, probe, loader, verbose=True, aggregate=True, writer=None, epoch=0):
    total_loss = 0.0
    metrics = None
    iteration = epoch * len(loader)

    with torch.inference_mode():
        pbar = tqdm(loader, desc="Evaluation") if verbose else loader
        for batch in pbar:
            images = batch["image"].cuda()
            mask = batch["depth"].cuda() > 0
            target = batch["snorm"].cuda()

            feats = model(images)
            feats = interpolate(feats, (120,120), mode="bilinear")
            pred = probe(feats)
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bicubic")

            writer.add_image('EVAL/pred', pred[0], iteration)
            writer.add_image('EVAL/gt', target[0], iteration)
            iteration += 1

            uncertainty = pred.shape[1] > 3
            loss = angular_loss(pred, target, mask, uncertainty_aware=uncertainty)

            total_loss += loss.item()
            batch_metrics = evaluate_surface_norm(pred.detach(), target, mask)
            if metrics is None:
                metrics = {key: [batch_metrics[key]] for key in batch_metrics}
            else:
                for key in batch_metrics:
                    metrics[key].append(batch_metrics[key])

    # aggregate
    total_loss = total_loss / len(loader)
    for key in metrics:
        metric_key = torch.cat(metrics[key], dim=0)
        metrics[key] = metric_key.mean() if aggregate else metric_key

    return total_loss, metrics


def train_model(rank, world_size, cfg):
    if world_size > 1:
        ddp_setup(rank, world_size, cfg.system.port)

    # ===== GET DATA LOADERS =====
    # validate and test on single gpu
    trainval_loader = build_loader(cfg.dataset, "trainval", cfg.batch_size, world_size)
    test_loader = build_loader(cfg.dataset, "test", cfg.batch_size, 1)
    trainval_loader.dataset.__getitem__(0)

    # ===== Get models =====
    model = instantiate(cfg.backbone)
    probe = instantiate(cfg.probe, feat_dim=model.feat_dim)

    # setup experiment name
    # === job info
    timestamp = datetime.now().strftime("%d%m%Y-%H%M")
    train_dset = trainval_loader.dataset.name
    test_dset = test_loader.dataset.name
    model_info = [
        f"{model.checkpoint_name:40s}",
        f"{model.patch_size:2d}",
        f"{str(model.layer):5s}",
        f"{model.output:10s}",
    ]
    probe_info = [f"{probe.name:25s}"]
    batch_size = cfg.batch_size * cfg.system.num_gpus
    train_info = [
        f"{cfg.optimizer.n_epochs:3d}",
        f"{cfg.optimizer.warmup_epochs:4.2f}",
        f"{cfg.optimizer.probe_lr:4.2e}",
        f"{cfg.optimizer.model_lr:4.2e}",
        f"{batch_size:4d}",
        f"{train_dset:10s}",
        f"{test_dset:10s}",
    ]
    # define exp_name
    exp_name = "_".join([timestamp] + model_info + probe_info + train_info +  [cfg.scheduler])
    exp_name = f"{exp_name}_{cfg.note}" if cfg.note != "" else exp_name
    exp_name = exp_name.replace(" ", "")  # remove spaces


    # ===== SETUP LOGGING =====
    log_path = cfg.log_path
    writer = SummaryWriter(f'{log_path}/{exp_name}')
    print('writer to logdir', writer.get_logdir())
    
    if rank == 0:
        exp_path = Path(__file__).parent / f"snorm_exps/{exp_name}"
        exp_path.mkdir(parents=True, exist_ok=True)
        logger.add(exp_path / "training.log")
        logger.info(f"Config: \n {OmegaConf.to_yaml(cfg)}")

    # move to cuda
    model = model.to(rank)
    probe = probe.to(rank)

    # very hacky ... SAM gets some issues with DDP finetuning
    model_name = model.checkpoint_name
    if "sam" in model_name or "vit-mae" in model_name:
        h, w = trainval_loader.dataset.__getitem__(0)["image"].shape[-2:]
        model.resize_pos_embed(image_size=(h, w))

    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        probe = DDP(probe, device_ids=[rank])

    if cfg.optimizer.model_lr == 0:
        optimizer = torch.optim.AdamW(
            [{"params": probe.parameters(), "lr": cfg.optimizer.probe_lr}]
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": probe.parameters(), "lr": cfg.optimizer.probe_lr},
                {"params": model.parameters(), "lr": cfg.optimizer.model_lr},
            ]
        )

    scheduler = get_scheduler(cfg, trainval_loader, optimizer)

    train(
        model,
        probe,
        trainval_loader,
        optimizer,
        scheduler,
        cfg.optimizer.n_epochs,
        detach_model=(cfg.optimizer.model_lr == 0),
        rank=rank,
        world_size=world_size,
        valid_loader=test_loader,
        writer=writer,
    )

    if rank == 0:
        logger.info(f"Evaluating on test split of {test_dset}")

        test_loss, test_metrics = validate(model, probe, test_loader, writer=writer, epoch=10*len(trainval_loader))
        logger.info(f"Final test loss       | {test_loss:.4f}")
        for metric in test_metrics:
            logger.info(f"Final test {metric:10s} | {test_metrics[metric]:.4f}")

        # result summary
        model_info = ", ".join(model_info)
        probe_info = ", ".join(probe_info)
        train_info = ", ".join(train_info)
        results = ", ".join([f"{test_metrics[_m]:.4f}" for _m in test_metrics])

        log = f"{timestamp}, {model_info}, {probe_info}, {train_info}, {results} \n"
        with open(f"snorm_results_{test_dset}.log", "a") as f:
            f.write(log)

        # save final model
        ckpt_path = exp_path / "ckpt.pth"
        checkpoint = {
            "cfg": cfg,
            "model": model.state_dict() if not isinstance(model, DDP) else model.module.state_dict(),
            "probe": probe.state_dict() if not isinstance(probe, DDP) else probe.module.state_dict(),
        }
        torch.save(checkpoint, ckpt_path)
        logger.info(f"Saved checkpoint at {ckpt_path}")

    if world_size > 1:
        destroy_process_group()


@hydra.main(config_name="snorm_training", config_path="./configs", version_base=None)
def main(cfg: DictConfig):
    world_size = cfg.system.num_gpus
    if world_size > 1:
        mp.spawn(train_model, args=(world_size, cfg), nprocs=world_size)
    else:
        train_model(0, world_size, cfg)


if __name__ == "__main__":
    main()