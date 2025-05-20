import math

import flax
from flax.training import train_state, checkpoints
import flax.training.checkpoints
import flax.training.train_state
import jax
import jax.numpy as jnp
import jax.random as random
from models import utils as mutils
from models import q_net
from dynamics import dynamics
import dynamics.utils as dutils
import optax
from typing import Any
import os

def get_loss(config, model, q_t, time_sampler, train):
  if 'am' == config.model.loss:
    loss_fn = get_am_loss(config, model, q_t, time_sampler, train)
  elif 'sam' == config.model.loss:
    loss_fn = get_stoch_am_loss(config, model, q_t, time_sampler, train)
  elif 'ssm' == config.model.loss:
    loss_fn = get_ssm_loss(config, model, q_t, time_sampler, train)
  elif 'dsm' == config.model.loss:
    loss_fn = get_dsm_loss(config, model, q_t, time_sampler, train)
  else:
    raise NotImplementedError(f'loss {config.model.loss} is not implemented')
  return loss_fn


def get_am_loss(config, model, q_t, time_sampler, train): # config, model, dynamics, time_sampler, train=True

  # if config.model.const_weight:
  #   w_t_fn = lambda t: jnp.ones_like(t)
  # else:

  CIFAR10_MEANS = jnp.array([0.4914, 0.4822, 0.4465])
  CIFAR10_STD = jnp.array([0.229, 0.224, 0.225])

  def rescale(img):
    img = img * 0.5 + 0.5 # обратно в [0, 1]
    img = (img - CIFAR10_MEANS) / CIFAR10_STD # в масштаб Q модели
    return img

  def get_states_actions(x_t, dsdx):
    x_t = rescale(x_t)
    scale = jnp.linspace(5, 10, num=dsdx.shape[0]).reshape((-1,1,1,1))
    dsdx /= scale
    dsdx = rescale(dsdx)
    return jnp.concatenate([x_t, dsdx], axis=-1)

  Q = q_net.RegressionInceptionNetV1()
  state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(os.path.dirname(__file__), "./Q_checkpoint"), target=None)

  class TrainState(train_state.TrainState):
    batch_stats: Any
  Q_state = TrainState.create(apply_fn=Q.apply,
                              params=state_dict["params"],
                              batch_stats=state_dict["batch_stats"],
                              tx=optax.identity())
  
  w_t_fn = lambda t: (1-t) # Модификация Action Matching (взвешенный)
  dwdt_fn = jax.grad(lambda t: w_t_fn(t).sum(), argnums=0)

  def am_loss(key, params, sampler_state, batch):
    keys = random.split(key, num=5)
    s = mutils.get_model_fn(model, params, train=train) # Функция модели f(t, x)
    dsdtdx_fn = jax.grad(lambda t,x,_key: s(t,x,_key).sum(), argnums=[0,1])
    
    data = batch['image']
    bs = data.shape[0]

    # sample time
    t_0, t_1 = config.data.t_0*jnp.ones((bs,1,1,1)), config.data.t_1*jnp.ones((bs,1,1,1))
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state) # sample_uniformly function
    order_idx = jnp.argsort(t)
    t = jnp.expand_dims(t, (1,2,3)) # [1, 2, 3] -> [ [[[1]]], [[[2]]], [[[3]]] ]
    # sample data
    x_0, x_1, x_t = q_t(keys[0], data, t, t_0, t_1) # (1 - t) * noise + t * data

    # boundaries loss
    s_0 = s(t_0, x_0, rng=keys[1])
    s_1 = s(t_1, x_1, rng=keys[2])
    loss = w_t_fn(t_0)*s_0.reshape((-1,1,1,1)) - w_t_fn(t_1)*s_1.reshape((-1,1,1,1))
    print(loss.shape, 'boundaries.shape')

    # time loss
    dsdt, dsdx = dsdtdx_fn(t, x_t, keys[3])
    p_t = time_sampler.invdensity(t)
    s_t = s(t, x_t, keys[4])
    print(p_t.shape, dsdt.shape, dsdx.shape, 'p_t.shape, dsdt.shape, dsdx.shape') # (bs, 1, 1, 1) (bs, 1, 1, 1) (bs, 32, 32, 3) p_t.shape, dsdt.shape, dsdx.shape
    loss += w_t_fn(t)*p_t*(dsdt + 0.5*(dsdx**2).sum((1,2,3), keepdims=True)) # Аппроксимация интеграла
    loss += s_t.reshape((-1,1,1,1))*dwdt_fn(t)*p_t # Производная сложной функции (из-за того, что домножаем на w(t))
    print(loss.shape, 'final.shape')

    states_actions = get_states_actions(x_t[order_idx], dsdx[order_idx])
    q_vals = Q.apply({"params": Q_state.params,
                      "batch_stats": Q_state.batch_stats},
                     states_actions,
                     train=False,
                     train_rng=None,
                     mutable=False)
    q_vals = jnp.clip(q_vals, min=0, max=20) # [0, 10]
    q_loss = jnp.sum(q_vals) / bs * 20 # примерно в диапазоне 0-400

    # order_idx = jnp.argsort(jnp.squeeze(t))
    # jax.debug.print('q_loss: {q_loss}, q_vals: {q_vals}', q_loss=q_loss, q_vals=q_vals)
    return loss.mean() - q_loss, (jnp.squeeze(q_vals)[order_idx], next_sampler_state) # mean - мат. ожидание в формуле

  return am_loss


def get_stoch_am_loss(config, model, q_t, time_sampler, train):

  w_t_fn = lambda t: (1-t)
  dwdt_fn = jax.grad(lambda t: w_t_fn(t).sum(), argnums=0)

  if config.model.anneal_sigma:
    sigma = lambda t: config.model.sigma * (1-t)
  else:
    sigma = lambda t: config.model.sigma

  def sam_loss(key, params, sampler_state, batch):
    keys = random.split(key, num=7)
    s = mutils.get_model_fn(model, params, train=train)
    dsdtdx_fn = jax.grad(lambda _t,_x,_key: s(_t,_x,_key).sum(), argnums=[0,1])
    
    data = batch['image']
    bs = data.shape[0]

    # sample time
    t_0, t_1 = config.data.t_0*jnp.ones((bs,1,1,1)), config.data.t_1*jnp.ones((bs,1,1,1))
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    t = jnp.expand_dims(t, (1,2,3))
    # sample data
    x_0, x_1, x_t = q_t(keys[0], data, t)

    # boundaries loss
    s_0 = s(t_0, x_0, rng=keys[1])
    s_1 = s(t_1, x_1, rng=keys[2])
    loss = w_t_fn(t_0)*s_0.reshape((-1,1,1,1)) - w_t_fn(t_1)*s_1.reshape((-1,1,1,1))
    print(loss.shape, 'boundaries.shape')

    # time loss
    eps = random.randint(keys[3], x_t.shape, 0, 2).astype(float)*2 - 1.0
    dsdx_val, jvp_val, dsdt_val = jax.jvp(lambda _x: dsdtdx_fn(t, _x, keys[4])[::-1], (x_t,), (eps,), has_aux=True)
    s_t = s(t, x_t, keys[5])
    p_t = time_sampler.invdensity(t)
    print(p_t.shape, dsdt_val.shape, dsdx_val.shape, 'p_t.shape, dsdt.shape, dsdx.shape')
    time_loss = dsdt_val + 0.5*(dsdx_val**2).sum((1,2,3), keepdims=True)
    time_loss += 0.5*sigma(t)**2*(jvp_val*eps).sum((1,2,3), keepdims=True)
    time_loss *= w_t_fn(t)
    time_loss += s_t.reshape((-1,1,1,1))*dwdt_fn(t)
    loss += p_t*time_loss
    print(loss.shape, 'final.shape')
    return loss.mean(), next_sampler_state

  return sam_loss


def get_ssm_loss(config, model, q_t, time_sampler, train):

  w_t_fn = lambda t: (1-t)**2

  def ssm_loss(key, params, sampler_state, batch):
    keys = random.split(key, num=7)
    s = mutils.get_model_fn(model, params, train=train)
    
    data = batch['image']
    bs = data.shape[0]

    # sample time
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    t = jnp.expand_dims(t, (1,2,3))
    # sample data
    _, _, x_t = q_t(keys[0], data, t)

    eps = random.randint(keys[3], x_t.shape, 0, 2).astype(float)*2 - 1.0
    s_val, jvp_val = jax.jvp(lambda _x: s(t, _x, keys[4]), (x_t,), (eps,))
    p_t = time_sampler.invdensity(t)
    print(p_t.shape, s_val.shape, jvp_val.shape, 'p_t.shape, s_val.shape, jvp_val.shape')
    loss = (jvp_val*eps).sum((1,2,3), keepdims=True) + 0.5*(s_val**2).sum((1,2,3), keepdims=True)
    loss *= w_t_fn(t)*p_t
    print(loss.shape, 'final.shape')
    return loss.mean(), next_sampler_state

  return ssm_loss


def get_dsm_loss(config, model, q_t, time_sampler, train):

  def dsm_loss(key, params, sampler_state, batch):
    keys = random.split(key, num=7)
    s = mutils.get_model_fn(model, params, train=train)
    
    data = batch['image']
    bs = data.shape[0]

    # sample time
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    t = jnp.expand_dims(t, (1,2,3))
    # sample data
    eps, _, x_t = q_t(keys[0], data, t)

    # eval loss
    loss = ((eps - s(t, x_t, keys[1])) ** 2).sum((1,2,3))
    print(loss.shape, 'final.shape')
    return loss.mean(), next_sampler_state

  return dsm_loss
