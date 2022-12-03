from examples.resnet.model import ResNet
from jax.random import PRNGKey
import jax.numpy as jnp

from prux import criterion, pruning, strategy

rng = PRNGKey(0)

model = ResNet(
    num_classes=10,
    dtype=jnp.float32,
    block_sizes=(2, 2, 2, 2),
    block_channels=(64, 128, 256, 512),
    block_strides=(1, 2, 2, 2),
)

params = model.init(rng, jnp.ones((1, 32, 32, 3)))

# %%

masks = criterion.RandomCriterion.get_masks(parameters=params, sparsity=0.2, globally=False, key=rng)
params = pruning.Pruning.apply(parameters=params, masks=masks, exclude=["BatchNorm"])
