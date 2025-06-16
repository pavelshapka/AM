from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import tensorflow as tf
import os

import run_lib

FLAGS = flags.FLAGS

# Считывание флагов запуска программы
config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval", "fid_stats"], "Running mode: train, eval or fid_stats")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config", "mode"])

def launch(argv):
  tf.config.experimental.set_visible_devices([], "GPU") # Отключение GPU для TensorFlow
  # tf.config.experimental.set_visible_devices([], "TPU") # Отключение TPU для TensorFlow

  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'

  if FLAGS.mode == "train":
    # Create the working directory
    # Run the training pipeline
    run_lib.train(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == "eval":
    # Run the evaluation pipeline
    run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
  elif FLAGS.mode == "fid_stats":
    # Run the evaluation pipeline
    run_lib.fid_stats(FLAGS.config, FLAGS.workdir)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(launch)
