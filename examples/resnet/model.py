import jax.numpy as jnp
import flax.linen as nn


class ResidualBlock(nn.Module):
    size: int
    channels: int
    stride: int = 1
    dtype: jnp.dtype = jnp.float32
    name: str = None
    train: bool = True

    @nn.compact
    def __call__(self, x, train=False):
        stride = self.stride

        for i in range(self.size):
            shortcut = x

            if self.channels != x.shape[-1] or stride != 1:
                shortcut = nn.Conv(self.channels, (1, 1), strides=(stride, stride), dtype=self.dtype)(x)

            x = nn.Conv(self.channels, (3, 3), stride)(x)
            x = nn.BatchNorm(use_running_average=True)(x)
            x = nn.relu(x)
            x = nn.Conv(self.channels, (3, 3))(x)
            x = nn.BatchNorm(use_running_average=True)(x)
            x = nn.relu(x + shortcut)
            stride = 1

        return x


class ResNet(nn.Module):
    num_classes: int
    dtype: jnp.dtype = jnp.float32
    block_sizes: list = (2, 2, 2, 2)
    block_channels: list = (64, 128, 256, 512)
    block_strides: list = (1, 2, 2, 2)

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.Conv(features=self.block_channels[0],
                    kernel_size=(5, 5),
                    padding='SAME',
                    use_bias=False,
                    dtype=self.dtype)(x)
        x = nn.BatchNorm(use_running_average=not train,
                         momentum=0.9,
                         dtype=self.dtype)(x)
        x = nn.relu(x)
        for i, (block_size, block_channels, block_stride) in enumerate(zip(self.block_sizes,
                                                                           self.block_channels,
                                                                           self.block_strides)):
            x = ResidualBlock(size=block_size,
                              channels=block_channels,
                              stride=block_stride,
                              dtype=self.dtype,
                              name=f'block_{i}')(x, train=train)
        x = nn.avg_pool(x, window_shape=x.shape[1:3], strides=(1, 1))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.num_classes, dtype=self.dtype)(x)
        return x
