import ml_collections


def get_config():
  config = ml_collections.ConfigDict()

  config.seed = 0

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'CIFAR10'
  data.ndims = 3 # Высота, ширина, каналы
  data.image_size = 32 # 32px x 32px
  data.uniform_dequantization = True # Предотвращает переобучение на целые числа
  data.num_channels = 3 # RGB
  data.random_flip = True # Случайный переворот изображения по горизонтали в процессе обучения для увеличения данных
  data.task = 'generate'
  data.dynamics = 'generation'
  data.t_0, data.t_1 = 0.0, 1.0

  # model
  config.model = model = ml_collections.ConfigDict()
  model.name = 'anet'
  model.loss = 'am'
  model.ema_rate = 0.9999 # Скорость экспоненциального скользящего среднего для обновления весов модели
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish' # SILU = x * sigmoid(x)
  model.nf = 128 # Базовое число фильтров (ядер) в модели
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,8)
  model.resamp_with_conv = True
  model.dropout = 0.1
  model.use_q_loss = True
  
  # training
  config.train = train = ml_collections.ConfigDict()
  train.batch_size = 128
  train.n_jitted_steps = 1
  train.n_iters = 500_000
  train.save_every = 2_500
  train.eval_every = 5_000
  train.log_every = 50
  train.lr = 1e-4
  train.beta1 = 0.9
  train.eps = 1e-8
  train.warmup = 5_000
  train.grad_clip = 1.
  train.euler_steps = 100
  train.q_loss_factor = 20

  # evaluation
  config.eval = eval = ml_collections.ConfigDict()
  eval.batch_size = 100
  eval.artifact_size = 64
  eval.num_samples = 50_000
  eval.use_ema = True
  eval.estimate_bpd = True
  eval.estimate_is_fid = True
  eval.euler_steps = 20

  return config