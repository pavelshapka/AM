import gc
import os
import functools
import json
import io

import wandb
from tqdm.auto import tqdm, trange

import jax
import flax
import numpy as np
import tensorflow as tf
# import tensorflow_gan as tfgan
import flax.jax_utils as flax_utils
from jax import random, jit
from jax import numpy as jnp
from flax.training import checkpoints
import math

import evaluation
import losses
import datasets
import train_utils as tutils
import eval_utils as eutils
from models import utils as mutils
from dynamics import utils as dutils
from models import anet, ddpm


def train(config, workdir):
  key = random.PRNGKey(config.seed) # фиксирование ключа генерации случайных чисел

  # init model
  key, init_key = random.split(key) # Разделение ключа на 2
  model, _, initial_params = mutils.init_model(init_key, config) # anet
  optimizer = tutils.get_optimizer(config) # Adam
  opt_state = optimizer.init(initial_params)

  # init dynamics
  dynamics = dutils.get_dynamics(config.data.dynamics) # generation: (1 - t) * noise + t * data
  time_sampler, init_sampler_state = dutils.get_time_sampler(config)

  state = mutils.State(step=1,
                       opt_state=opt_state,
                       model_params=initial_params,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       sampler_state=init_sampler_state,
                       key=key, wandbid=np.random.randint(int(1e7),int(1e8)))

  checkpoint_dir = os.path.join(workdir, "checkpoints")
  tf.io.gfile.makedirs(checkpoint_dir)
  state = checkpoints.restore_checkpoint(checkpoint_dir, state)
  initial_step = int(state.step)
  key = state.key

  if jax.process_index() == 0:
    wandb.init(id=str(state.wandbid), 
                project=config.data.task + '_' + config.data.dataset, 
                resume="allow",
                config=json.loads(config.to_json_best_effort()))
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = str(state.wandbid)
  
  # init train step
  loss_fn = losses.get_loss(config, model, dynamics, time_sampler, train=True)
  print(loss_fn)
  wandb.log(dict(loss=loss_fn), step=0)
  step_fn = tutils.get_step_fn(optimizer, loss_fn)
  p_step_fn = jax.pmap(functools.partial(jax.lax.scan, step_fn), axis_name='batch', donate_argnums=1) # batch - специальная ось для выполнения групповых операций на данных, расположенных на разных устройствах

  # artifacts init
  artifact_shape = (config.eval.artifact_size, 
                    config.data.image_size, 
                    config.data.image_size, 
                    config.data.num_channels)
  pshape = (jax.local_device_count(), artifact_shape[0]//jax.local_device_count()) + artifact_shape[1:]
  artifact_generator, trajectory_generator = tutils.get_artifact_generator(model, config, dynamics, pshape[1:], is_dopri5=False)
  p_artifact_generator = jax.pmap(artifact_generator, axis_name='batch')
  p_trajectory_generator = jax.pmap(trajectory_generator, axis_name='batch')

  # init dataloaders
  train_ds, _, _ = datasets.get_dataset(config, 
                                        additional_dim=config.train.n_jitted_steps, # количество итераций в рамках одного шага в цикле for
                                        uniform_dequantization=config.data.uniform_dequantization)
  train_iter = iter(train_ds)
  scaler = datasets.get_image_scaler(config)
  inverse_scaler = datasets.get_image_inverse_scaler(config)

  # run train
  assert (config.train.n_iters % config.train.save_every) == 0

  pstate = flax_utils.replicate(state) # Распределение состояния по всем устройствам
  key = jax.random.fold_in(key, jax.process_index()) # Обновление ключа в зависимости от индекса процесса
  for step in range(initial_step, config.train.n_iters+1, config.train.n_jitted_steps):
    batch = jax.tree_map(lambda x: scaler(x._numpy()), next(train_iter)) # Достаем батч и применяем функцию масштабирования к каждому элементу батча
    key, *next_key = random.split(key, num=jax.local_device_count() + 1)
    next_key = jnp.asarray(next_key)
    (_, pstate), (ploss, pq_vals) = p_step_fn((next_key, pstate), batch) # ploss: (num_devices, n_jitted_steps)
    loss = ploss.mean() # средний loss по n_jitted_steps
    # q_vals = pq_vals.mean() # средний q_vals по n_jitted_steps
    if (step % (config.train.log_every * 20) == 0) and (jax.process_index() == 0):
      log_dict = dict(loss=loss.item())
      if config.model.use_q_loss:
        log_dict['q_vals'] = jnp.sum(pq_vals)
        # with open("q_vals.txt", "a") as f:
        #   f.write(f"{step} => {jnp.sum(pq_vals)} {pq_vals}\n")

      wandb.log(log_dict, step=step)

    if (step % config.train.save_every == 0) and (jax.process_index() == 0):
      saved_state = flax_utils.unreplicate(pstate)
      saved_state = saved_state.replace(key=key)
      checkpoints.save_checkpoint(checkpoint_dir, saved_state,
                                  step=step // config.train.save_every,
                                  keep=50)

    if step % config.train.eval_every == 0:
      data = batch['image'][:,0].reshape((-1,) + artifact_shape[1:])
      data = data[:artifact_shape[0]].reshape(pshape)
      key, *next_keys = random.split(key, num=jax.local_device_count() + 1)
      next_keys = jnp.asarray(next_keys)
      artifacts, num_steps = p_artifact_generator(next_keys, pstate, data)
      print(artifacts.shape, num_steps)
      final_x = inverse_scaler(artifacts.reshape(artifact_shape))
      wandb.log(dict(examples=[wandb.Image(tutils.stack_imgs(final_x))],
                     nfe=jnp.mean(num_steps).item()), step=step)

def evaluate(config, workdir, eval_folder):
  key = random.PRNGKey(config.seed)
  
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)
  sample_dir = os.path.join(eval_dir, 'samples')
  tf.io.gfile.makedirs(sample_dir)
  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # init model
  key, init_key = random.split(key)
  model, _, initial_params = mutils.init_model(init_key, config)

  # init dynamics
  dynamics = dutils.get_dynamics(config.data.dynamics)

  state = mutils.State(step=0, 
                       opt_state=None,
                       model_params=initial_params,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       sampler_state=None,
                       key=key, wandbid=np.random.randint(int(1e7),int(1e8)))
  state = checkpoints.restore_checkpoint(checkpoint_dir, state)

  eval_state = eutils.EvalState(bpd_batch_id=0,
                                sample_batch_id=0,
                                global_iter=0,
                                key=key)
  eval_state = checkpoints.restore_checkpoint(eval_dir, eval_state, prefix='eval_state_')

  # init image generator
  artifact_shape = (config.eval.batch_size, 
                    config.data.image_size, 
                    config.data.image_size, 
                    config.data.num_channels)
  pshape = (jax.local_device_count(), artifact_shape[0]//jax.local_device_count()) + artifact_shape[1:]
  artifact_generator = eutils.get_artifact_generator(model, config, dynamics, pshape[1:], is_dopri5=False)
  p_artifact_generator = jax.pmap(artifact_generator, axis_name='batch')

  # init dataloaders
  train_ds, test_ds, _ = datasets.get_dataset(config, additional_dim=None,
                                              uniform_dequantization=False,
                                              evaluation=True)
  train_iter, test_iter = iter(train_ds), iter(test_ds)
  scaler = datasets.get_image_scaler(config)
  inverse_scaler = datasets.get_image_inverse_scaler(config)

  bpds = []
  global_iter = eval_state.global_iter
  if config.eval.estimate_bpd:
    # init bpd estimator
    get_bpd = eutils.get_bpd_estimator(model, config, is_dopri5=False)
    p_get_bpd = jax.pmap(get_bpd, axis_name='batch')

    # estimate bpd
    pstate = flax_utils.replicate(state)
    key = jax.random.fold_in(key, jax.process_index())
    for _ in range(eval_state.bpd_batch_id):
      next(test_iter)
    for batch_id in range(eval_state.bpd_batch_id, len(test_ds)):
      batch = jax.tree_map(lambda x: scaler(x._numpy()), next(test_iter))
      key, *next_key = random.split(key, num=jax.local_device_count() + 1)
      next_key = jnp.asarray(next_key)
      bpd, num_steps = p_get_bpd(next_key, pstate, batch)
      bpds.append(bpd)
      print(f'batch {batch_id}/{len(test_ds)}, bpd: {bpd}, num_steps: {num_steps}')
      global_iter += 1
      eval_state = eval_state.replace(bpd_batch_id=batch_id, key=key, global_iter=global_iter)
      checkpoints.save_checkpoint(eval_dir, eval_state, step=global_iter, keep=1, prefix='eval_state_')

  if len(bpds) == 0:
    bpds.append(-jnp.ones(1))
  bpds = jnp.stack(bpds)
  mean_bpd, std_bpd = bpds.mean(), bpds.std()
  mean_bpd, std_bpd = 0, 0
  print(f'final bpd: {mean_bpd}/{std_bpd}')

  # init inception
  inception_model = evaluation.get_inception_model(inceptionv3=False)

  # generate samples
  pstate = flax_utils.replicate(state) # реплицируем состояние на устройства
  num_batches = math.ceil(config.eval.num_samples / config.eval.batch_size) # колмчество батчей, необходимых для генерации заданного количества сэмплов
  print(eval_state.sample_batch_id, num_batches)
  for _ in range(eval_state.sample_batch_id): # пропускаем уже обработанные батчи
    next(train_iter)
  for batch_id in range(eval_state.sample_batch_id, num_batches):
    batch = jax.tree_map(lambda x: scaler(x._numpy()), next(train_iter)) # достаем батч и скейлим [0, 1] -> [-1, 1]
    key, *next_keys = random.split(key, num=jax.local_device_count() + 1)
    next_keys = jnp.asarray(next_keys)
    artifacts, num_steps = p_artifact_generator(next_keys, pstate, batch['image']) # генерируем картинку
    artifacts = inverse_scaler(artifacts) # скейлим обратно
    artifacts = artifacts.reshape(artifact_shape)
    artifacts = jnp.clip(artifacts*255.0, 0.0, 255.0).astype(np.uint8) # приводим к RGB виду
    print(f'{batch_id}/{len(train_ds)}, artifacts.shape: {artifacts.shape}, num_steps: {num_steps}')
    with tf.io.gfile.GFile(os.path.join(sample_dir, f"samples_{batch_id}.npz"), "wb") as fout: # сохраняем сэмплы в файл
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, samples=artifacts, num_steps=num_steps)
          fout.write(io_buffer.getvalue())

    gc.collect()
    latents = evaluation.run_inception_distributed(artifacts, inception_model) # вычисляем латентные представления модели Inception
    gc.collect()
    with tf.io.gfile.GFile(os.path.join(sample_dir, f"statistics_{batch_id}.npz"), "wb") as fout:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, pool_3=latents["pool_3"], logits=latents["logits"]) # pool_3 - название одного из слоев в Inception
      fout.write(io_buffer.getvalue())
    global_iter += 1
    eval_state = eval_state.replace(bpd_batch_id=batch_id, key=key, global_iter=global_iter)
    checkpoints.save_checkpoint(eval_dir, eval_state, step=global_iter, keep=1, prefix='eval_state_')
  
  all_logits = []
  all_pools = []
  stats = tf.io.gfile.glob(os.path.join(sample_dir, "statistics_*.npz")) # читаются все такие файлы и заполняются списки all_logits и all_pools
  for stat_file in stats:
    with tf.io.gfile.GFile(stat_file, "rb") as fin:
      stat = np.load(fin)
      all_logits.append(stat["logits"])
      all_pools.append(stat["pool_3"])
  all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples] # Объедиенение в 1 массив
  all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

  data_stats = evaluation.load_dataset_stats(config) # Загружаются статистики реальных изображений (например, средние значения и дисперсии признаков pool_3)
  data_pools = data_stats["pool_3"][:config.eval.num_samples]

  inception_score = evaluation.calculate_inception_score(all_logits)
  fid = evaluation.calculate_frechet_distance(data_pools, all_pools)
  print(f'IS: {inception_score}, FID: {fid}')
  
  with tf.io.gfile.GFile(os.path.join(eval_dir, f"report.npz"), "wb") as f:
    io_buffer = io.BytesIO()
    np.savez_compressed(io_buffer, IS=inception_score, fid=fid, mean_bpd=mean_bpd, std_bpd=std_bpd)
    f.write(io_buffer.getvalue())


def fid_stats(config, workdir, fid_folder="assets/stats"):
  """Evaluate trained models.
  Args:
    config: Configuration to use.
    fid_folder: The subfolder for storing fid statistics. 
  """
  # Create directory to eval_folder
  fid_dir = os.path.join(workdir, fid_folder)
  tf.io.gfile.makedirs(fid_dir)

  # Build data pipeline
  train_ds, _, dataset_builder = datasets.get_dataset(config,
                                                      additional_dim=None,
                                                      uniform_dequantization=False,
                                                      evaluation=True)
  train_iter = iter(train_ds)

  inception_model = evaluation.get_inception_model(inceptionv3=False)

  all_pools = []
  num_batches = math.ceil(config.eval.num_samples / config.eval.batch_size)
  for batch_id in range(num_batches):
    batch = next(train_iter)
    print("Making FID stats -- step: %d" % (batch_id))
    batch_ = jax.tree_map(lambda x: x._numpy(), batch)
    batch_ = (batch_['image']*255).astype(np.uint8).reshape((-1, config.data.image_size, config.data.image_size, 3))

    # Force garbage collection before calling TensorFlow code for Inception network
    gc.collect()
    latents = evaluation.run_inception_distributed(batch_, inception_model)
    all_pools.append(latents["pool_3"])
    # Force garbage collection again before returning to JAX code
    gc.collect()

  all_pools = np.concatenate(all_pools, axis=0) # Combine into one

  # Save latent represents of the Inception network to disk or Google Cloud Storage
  filename = f'{config.data.dataset.lower()}_stats.npz'
  with tf.io.gfile.GFile(os.path.join(fid_dir, filename), "wb") as fout:
    io_buffer = io.BytesIO()
    np.savez_compressed(io_buffer, pool_3=all_pools)
    fout.write(io_buffer.getvalue())
