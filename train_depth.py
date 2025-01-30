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
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.distributed import destroy_process_group, init_process_group


from torch.nn.functional import interpolate
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR, ConstantLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from evals.datasets.builder import build_loader
from evals.utils.losses import DepthLoss, DepthLossV2, DepthLossV3, DepthSigLoss, L1LogLoss, L1Loss
from evals.utils.metrics import evaluate_depth, match_scale_and_shift
from evals.utils.optim import cosine_decay_linear_warmup, get_cosine_schedule_with_warmup_LambdaLR, get_scheduler, linear_decay_lr


def ddp_setup(rank: int, world_size: int, port: int):
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
    loss_fn,
    rank=0,
    world_size=1,
    valid_loader=None,
    writer=None,
    scale_invariant=False,
):
    for ep in range(n_epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(ep)

        train_loss = 0
        pbar = tqdm(train_loader) if rank == 0 else train_loader
        for i, batch in enumerate(pbar):
            images = batch["image"].to(rank)
            target = batch["depth"].to(rank)
            optimizer.zero_grad()
            if detach_model:
                with torch.no_grad():
                    feats = model(images)
                    if isinstance(feats, (tuple, list)):
                        feats = [_f.detach() for _f in feats]
                    else:
                        feats = feats.detach()
            else:
                feats = model(images)

            pred = probe(feats)
            pred = interpolate(pred, size=target.shape[-2:], mode="bilinear")

            if scale_invariant:
                pred = match_scale_and_shift(pred, target)
                pred = pred.clamp(min=0.001, max=10.0)

            if i % 100 == 0:
                writer.add_image('LOSS/pred', pred[0] / 10, ep * len(train_loader) + i)
                writer.add_image('LOSS/gt', target[0] / 10, ep * len(train_loader) + i)

            loss, loss_dict = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            pr_lr = optimizer.param_groups[0]["lr"]
            loss = loss.item()
            train_loss += loss

            if rank == 0:
                _loss = train_loss / (i + 1)
                pbar.set_description(
                    f"{ep} | loss: {loss:.4f} ({_loss:.4f}) probe_lr: {pr_lr:.2e}"
                )
                if i % 100 == 0:
                    writer.add_scalar(f'Accumulated_Loss', _loss, ep * len(train_loader) + i)
                    writer.add_scalar(f'Loss', loss, ep * len(train_loader) + i)
                    for key, value in loss_dict.items():
                        writer.add_scalar(f'Loss_{key}', value, ep * len(train_loader) + i)
                    writer.add_scalar(f'probe_lr', pr_lr, ep * len(train_loader) + i)


        train_loss /= len(train_loader)

        if rank == 0:
            logger.info(f"train loss {ep}   | {train_loss:.4f}")
            if valid_loader is not None:
                print('Evaluate:', len(valid_loader))
                val_loss, val_metrics = validate(
                    model, probe, valid_loader, loss_fn, scale_invariant=False,writer=writer, epoch= ep
                )
                logger.info(f"valid loss {ep}   | {val_loss:.4f}")
                writer.add_scalar(f'Eval/loss', val_loss, (ep + 1) * len(train_loader))
                for metric in val_metrics:
                    logger.info(f"valid Scale Aware {metric:10s} | {val_metrics[metric]:.4f}")
                    writer.add_scalar(f'Eval/{metric}', val_metrics[metric], (ep+1) * len(train_loader))
                _ , val_metrics = validate(
                    model, probe, valid_loader, loss_fn, scale_invariant=False,writer=writer, epoch= ep
                )
                for metric in val_metrics:
                    logger.info(f"valid Scale Invariant {metric:10s} | {val_metrics[metric]:.4f}")
                    writer.add_scalar(f'Eval/{metric}', val_metrics[metric], (ep+1) * len(train_loader))

def validate(
    model, probe, loader, loss_fn, verbose=True, scale_invariant=False, aggregate=True, writer=None, epoch=0
):
    total_loss = 0.0
    metrics = None
    iteration = epoch * len(loader)
    with torch.inference_mode():
        pbar = tqdm(loader, desc="Evaluation") if verbose else loader
        for batch in pbar:
            images = batch["image"].cuda()
            target = batch["depth"].cuda()

            feat = model(images)
            pred = probe(feat).detach()
            pred = interpolate(pred, size=target.shape[-2:], mode="bilinear")

            writer.add_image('EVAL/pred', pred[0] / 10 , iteration)
            writer.add_image('EVAL/gt', target[0] / 10, iteration)
            iteration += 1
            
            loss, _ = loss_fn(pred, target)
            total_loss += loss.item()

            batch_metrics = evaluate_depth(
                pred, target, scale_invariant=scale_invariant
            )
            if metrics is None:
                metrics = {
                    key: [
                        value,
                    ]
                    for key, value in batch_metrics.items()
                }
            else:
                for key, value in batch_metrics.items():
                    metrics[key].append(value)

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
    print('loading')
    #checkpoint = torch.load("depth_exps/07102024-1905_vit-mae-base_16_2-5-8-11_dense_bindepth_dpt_k3_10_1.50_0.0005_0.0_8_NYUv2_NYUv2/ckpt.pth")
    #"depth_exps/03102024-2156_snapshot_194_0.524.pth_16_last_dense_bindepth_linear_k3_10_1.50_0.0005_0.0_8_NYUv2_NYUv2/ckpt.pth")
    #cfg = checkpoint['cfg']
    print('cfg', cfg.backbone)
    model = instantiate(cfg.backbone)
    print('MODEL OUTFEATURESIZE 120')
    probe = instantiate(
        cfg.probe, feat_dim=model.feat_dim, max_depth=trainval_loader.dataset.max_depth
    )

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
        f"{str(cfg.optimizer.probe_lr):>10s}",
        f"{str(cfg.optimizer.model_lr):>10s}",
        f"{batch_size:4d}",
        f"{train_dset:10s}",
        f"{test_dset:10s}",
    ]

    # define exp_name
    exp_name = "_".join([timestamp] + model_info + probe_info + train_info + [cfg.loss] + [cfg.scheduler])
    exp_name = f"{exp_name}_{cfg.note}" if cfg.note != "" else exp_name
    exp_name = exp_name.replace(" ", "")  # remove spaces

    # ===== SETUP LOGGING =====
    log_path = cfg.log_path
    writer = SummaryWriter(f'{log_path}/{exp_name}')
    print('writer to logdir', writer.get_logdir())

    if rank == 0:
        exp_path = Path(__file__).parent / f"{log_path}/{exp_name}"
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

    # move to DDP
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
    loss_fn = get_loss_function(cfg)

    train(
        model,
        probe,
        trainval_loader,
        optimizer,
        scheduler,
        cfg.optimizer.n_epochs,
        detach_model=(cfg.optimizer.model_lr == 0),
        loss_fn=loss_fn,
        rank=rank,
        world_size=world_size,
        valid_loader=test_loader,
        writer= writer,
        scale_invariant=cfg.scale_invariant
    )

    if rank == 0:
        logger.info(f"Evaluating on test split of {test_dset}")

        test_sa_loss, test_sa_metrics = validate(model, probe, test_loader, loss_fn, writer=writer, epoch=10*len(trainval_loader))
        logger.info(f"Scale-Aware Final test loss       | {test_sa_loss:.4f}")
        for metric in test_sa_metrics:
            logger.info(f"Final test SA {metric:10s} | {test_sa_metrics[metric]:.4f}")
        results_sa = ", ".join([f"{test_sa_metrics[_m]:.4f}" for _m in test_sa_metrics])

        # get scale invariant
        test_si_loss, test_si_metrics = validate(
            model, probe, test_loader, loss_fn, scale_invariant=True, writer=writer, epoch=10*len(trainval_loader)
        )
        logger.info(f"Scale-Invariant Final test loss       | {test_si_loss:.4f}")
        for metric in test_si_metrics:
            logger.info(f"Final test SI {metric:10s} | {test_si_metrics[metric]:.4f}")
        results_si = ", ".join([f"{test_si_metrics[_m]:.4f}" for _m in test_si_metrics])

        # log experiments
        exp_info = ", ".join(model_info + probe_info + train_info)
        log = f"{timestamp}, {exp_info}, {results_sa}, {results_si} \n"
        with open(f"depth_results_{test_dset}.log", "a") as f:
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

def get_loss_function(cfg):
    if cfg.loss == "DepthLoss":
        return DepthLoss()
    elif cfg.loss == "L1LogLoss":
        return L1LogLoss()
    elif cfg.loss == "DepthLossV2":
        return DepthLossV2()
    elif cfg.loss == "DepthLossV3":
        return DepthLossV3()
    elif cfg.loss == "DepthSigLoss":
        return DepthSigLoss()
    elif cfg.loss == "L1Loss":
        return L1Loss()
    else:
        raise ValueError(f"Loss {cfg.loss} not supported")


@hydra.main(config_name="depth_training", config_path="./configs", version_base=None)
def main(cfg: DictConfig):
    world_size = cfg.system.num_gpus
    if world_size > 1:
        mp.spawn(train_model, args=(world_size, cfg), nprocs=world_size)
    else:
        train_model(0, world_size, cfg)


if __name__ == "__main__":
    main()
