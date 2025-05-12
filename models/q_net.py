from typing import Callable

import functools

import jax.numpy as jnp
from flax import linen as nn


class ConvNxN(nn.Module):
    N: int
    out_channels: int
    stride: int = 1
    padding: str = "SAME"
    activation: Callable = nn.relu
    use_bias: bool = True

    name: str = "ConvNxN"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        x = nn.Conv(features=self.out_channels,
                    kernel_size=(self.N, self.N),
                    kernel_init=nn.initializers.kaiming_normal(),
                    strides=(self.stride, self.stride),
                    padding=self.padding,
                    use_bias=self.use_bias,
                    name="ConvNxN")(x)
        x = nn.BatchNorm(use_running_average=not train,
                         name="BatchNorm")(x)
        x = self.activation(x)
        return x

class InceptionInput(nn.Module):
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        convNxN = functools.partial(ConvNxN, activation=self.activation, padding="SAME", use_bias=False)

        x = convNxN(N=3,
                    out_channels=64,
                    stride=1,
                    name="conv_1_3x3_1")(x, train)
        x = convNxN(N=3,
                    out_channels=64,
                    stride=1,
                    name="conv_2a_3x3_1")(x, train)
        x = convNxN(N=3,
                    out_channels=192,
                    stride=1,
                    name="conv_2b_3x3_1")(x, train)
        return x

class InceptionBlock(nn.Module):
    out_channels: dict[str, int]
    reduced_channels: dict[str, int]
    activation: Callable = nn.relu
    padding: str = "SAME"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        conv_1x1 = functools.partial(ConvNxN,
                                     N=1,
                                     activation=self.activation,
                                     padding=self.padding,
                                     use_bias=False)
        conv_3x3 = functools.partial(ConvNxN,
                                     N=3,
                                     activation=self.activation,
                                     padding=self.padding,
                                     use_bias=False)
        conv_5x5 = functools.partial(ConvNxN,
                                     N=5,
                                     activation=self.activation,
                                     padding=self.padding,
                                     use_bias=False)
                                     
        x_1x1 = conv_1x1(out_channels=self.out_channels["conv1x1"], stride=1, name="conv_a_1x1_1")(x, train)
        
        x_3x3 = conv_1x1(out_channels=self.reduced_channels["conv3x3"], stride=1, name="conv_b_3x3_1")(x, train)
        x_3x3 = conv_3x3(out_channels=self.out_channels["conv3x3"], stride=1, name="conv_b_3x3_2")(x_3x3, train)
        
        x_5x5 = conv_1x1(out_channels=self.reduced_channels["conv5x5"], stride=1, name="conv_c_5x5_1")(x, train)
        x_5x5 = conv_5x5(out_channels=self.out_channels["conv5x5"], stride=1, name="conv_c_5x5_2")(x_5x5, train)
        
        x_mp = nn.max_pool(inputs=x,
                           window_shape=(3, 3),
                           strides=(1, 1),
                           padding="SAME")
        x_mp = conv_1x1(out_channels=self.out_channels["max_pool"], stride=1, name="conv_d_1x1_1")(x_mp, train)
        
        x = jnp.concatenate([x_1x1, x_3x3, x_5x5, x_mp], axis=-1)
        return x


class RegressionInceptionNetV1(nn.Module):
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 train: bool = False,
                 train_rng: jnp.ndarray = None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        max_pool = functools.partial(nn.max_pool, padding="SAME")

        x = InceptionInput(activation=self.activation)(x, train)
        
        x = InceptionBlock(out_channels={"conv1x1": 64, "conv3x3": 128, "conv5x5": 32, "max_pool": 32},
                           reduced_channels={"conv3x3": 96, "conv5x5": 16},
                           activation=self.activation)(x, train)
        x = InceptionBlock(out_channels={"conv1x1": 128, "conv3x3": 192, "conv5x5": 96, "max_pool": 64},
                           reduced_channels={"conv3x3": 128, "conv5x5": 32},
                           activation=self.activation)(x, train)
        x = max_pool(inputs=x,
                     window_shape=(3, 3),
                     strides=(2, 2))
        x = InceptionBlock(out_channels={"conv1x1": 192, "conv3x3": 208, "conv5x5": 48, "max_pool": 64},
                           reduced_channels={"conv3x3": 96, "conv5x5": 16},
                           activation=self.activation)(x, train)
        
        x = InceptionBlock(out_channels={"conv1x1": 160, "conv3x3": 224, "conv5x5": 64, "max_pool": 64},
                           reduced_channels={"conv3x3": 112, "conv5x5": 24},
                           activation=self.activation)(x, train)
        x = InceptionBlock(out_channels={"conv1x1": 128, "conv3x3": 256, "conv5x5": 64, "max_pool": 64},
                           reduced_channels={"conv3x3": 128, "conv5x5": 24},
                           activation=self.activation)(x, train)
        x = InceptionBlock(out_channels={"conv1x1": 112, "conv3x3": 288, "conv5x5": 64, "max_pool": 64},
                           reduced_channels={"conv3x3": 144, "conv5x5": 32},
                           activation=self.activation)(x, train)
        
        x = InceptionBlock(out_channels={"conv1x1": 256, "conv3x3": 320, "conv5x5": 128, "max_pool": 128},
                           reduced_channels={"conv3x3": 160, "conv5x5": 32},
                           activation=self.activation)(x, train)
        x = max_pool(inputs=x,
                     window_shape=(3, 3),
                     strides=(2, 2))
        x = InceptionBlock(out_channels={"conv1x1": 256, "conv3x3": 320, "conv5x5": 128, "max_pool": 128},
                    reduced_channels={"conv3x3": 160, "conv5x5": 32},
                    activation=self.activation)(x, train)
        x = InceptionBlock(out_channels={"conv1x1": 384, "conv3x3": 384, "conv5x5": 128, "max_pool": 128},
                    reduced_channels={"conv3x3": 192, "conv5x5": 48},
                    activation=self.activation)(x, train)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dropout(rate=0.4)(x, deterministic=not train, rng=train_rng)
        x = nn.Dense(features=1,
                     kernel_init=nn.initializers.kaiming_normal(),
                     use_bias=False)(x)

        return x
