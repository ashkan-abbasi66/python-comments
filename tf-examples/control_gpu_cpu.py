# NUM_CORES = 1
# inter_op_parallelism_threads=NUM_CORES, # add these lines to CongigProto
# intra_op_parallelism_threads=NUM_CORES

# device_count={'GPU': 0}
session_conf = tf.ConfigProto(
      device_count={'CPU': 1, 'GPU': 0},
      allow_soft_placement=True,
      log_device_placement=False
      )


# GPU ALLOCATION
# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# sess=tf.Session()
sess = tf.Session(config=session_conf)