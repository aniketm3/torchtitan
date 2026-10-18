# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Throughput sweep recipes (Aleph Alpha "hierarchical approach" reproduction).

Each sweep point is a config function whose name encodes it, so the standard
loader selects it without extra flags:

    MODULE=torchtitan_recipes.aleph_alpha_scaling.recipes \\
    CONFIG=aa_1p7b_selective_lbs8_fsdp1_dp1 NGPU=1 ./run_train.sh

Name format: ``aa_{model}_{ac}_lbs{LBS}_fsdp{FSDP}_dp{DP}[_ep8][_eager][_log10][_prof]``

Optional suffixes, one change each:
- ``_ep8``     -- (MoE only) expert parallelism over 8 GPUs: each GPU owns 16 of
                  the 128 experts, and tokens travel to their experts' GPU via
                  all-to-all. EP borrows its ranks from FSDP, so it needs fsdp=8.
- ``_eager``   -- turn off torch.compile. By default Qwen3 compiles a few small
                  regions (loss, SwiGLU activation, RoPE, fp32->bf16 split);
                  matmuls and attention are separate kernels either way.
- ``_log10``   -- log metrics every 10 steps instead of every step. Logging calls
                  ``.item()``, which syncs CPU and GPU; per-step logging leaves
                  the GPU idle while the CPU prepares the next batch.
- ``_prof``    -- capture a PyTorch profiler trace of steps 19-20 and a CUDA
                  memory snapshot after step 5.

model:
- ``1p7b``     -- Qwen3 1.7B dense (1.72B params; embedding and LM head are tied)
- ``30b_a3b``  -- Qwen3 30B-A3B MoE

ac (activation checkpointing):
- ``none``      -- store every activation, recompute nothing.
- ``selective`` -- store the outputs of the expensive ops (attention, linears);
                   recompute the cheap ones (norms, SiLU, RoPE) in backward.
- ``full``      -- store only each block's input; recompute the whole block.
- ``default``   -- torchtitan's default policy (selective, but MoE models also
                   recompute the routed-expert w13 GEMM). Not in the blog.

Fixed for every run: seq_len 4096, bf16 compute (fp32 master weights), local
c4_test data (repeats; no streaming), Qwen3's default compiled regions,
gradient accumulation 1, NUM_STEPS steps.
"""

import re

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import (
    ActivationRematConfig,
    DefaultActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3 import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer

from torchtitan_recipes.data import c4_document_capacity

SEQ_LEN = 4096
NUM_STEPS = 40  # summarize.py discards the first 10 as warmup

# name in config -> (flavor in torchtitan/models/qwen3/flavors.py, tokenizer dir)
MODELS = {
    "1p7b": ("1.7B", "./assets/hf/Qwen3-1.7B"),
    "30b_a3b": ("30B-A3B", "./assets/hf/Qwen3-30B-A3B"),
}

_CONFIG_NAME_RE = re.compile(
    r"aa_(1p7b|30b_a3b)_(none|selective|full|default)_lbs(\d+)_fsdp(\d+)_dp(\d+)"
    r"(_ep8)?(_eager)?(_log10)?(_prof)?"
)


def _activation_remat(ac: str) -> ActivationRematConfig | None:
    if ac == "none":
        return None
    if ac == "selective":
        # Keep every model-declared region (attention, linears, expert GEMMs);
        # everything outside a region (norms, activations, RoPE) is recomputed.
        return ActivationRematConfig(save_regions=["*"])
    if ac == "full":
        return FullActivationRematConfig()
    if ac == "default":
        return DefaultActivationRematConfig()
    raise ValueError(f"Unknown AC mode {ac!r}.")


def build_sweep_config(
    model: str,
    ac: str,
    *,
    local_batch_size: int,
    fsdp: int,
    dp: int,
    ep: int = 1,
    eager: bool = False,
    log_freq: int = 1,
    profile: bool = False,
) -> Trainer.Config:
    """Build the config for one sweep point."""
    flavor, hf_assets_path = MODELS[model]
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=fsdp,
        data_parallel_replicate_degree=dp,
        expert_parallel_degree=ep,
        tensor_parallel_degree=1,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
    )
    model_config = build_model_config(flavor, seq_len=SEQ_LEN, parallelism=parallelism)
    if eager:
        model_config.local_compile_regions = []
    num_tokens_per_microbatch_per_dp_rank = local_batch_size * SEQ_LEN
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path=hf_assets_path,
        model=model_config,
        dataloader=GrainDataLoader.Config(
            max_num_documents=c4_document_capacity(
                num_tokens_per_microbatch_per_dp_rank
            ),
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=5),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch_per_dp_rank,
            max_context_length=SEQ_LEN,
            steps=NUM_STEPS,
        ),
        # TensorBoard records per-step data-loading time, which the console omits.
        metrics=MetricsProcessor.Config(log_freq=log_freq, enable_tensorboard=True),
        profiler=Profiler.Config(
            enable_profiling=profile,
            profile_freq=20,
            profiler_warmup=3,
            profiler_active=2,
            profiler_repeat=1,
            enable_memory_snapshot=profile,
            memory_snapshot_freq=5,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=_activation_remat(ac),
    )


def __getattr__(name: str):
    match = _CONFIG_NAME_RE.fullmatch(name)
    if match is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    model, ac, lbs, fsdp, dp, ep_suffix, eager_suffix, log10_suffix, prof_suffix = (
        match.groups()
    )

    def config_fn() -> Trainer.Config:
        return build_sweep_config(
            model,
            ac,
            local_batch_size=int(lbs),
            fsdp=int(fsdp),
            dp=int(dp),
            ep=8 if ep_suffix is not None else 1,
            eager=eager_suffix is not None,
            log_freq=10 if log10_suffix is not None else 1,
            profile=prof_suffix is not None,
        )

    config_fn.__name__ = name
    return config_fn
