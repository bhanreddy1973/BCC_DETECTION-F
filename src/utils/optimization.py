import tensorflow as tf
import os
import psutil

def enable_mixed_precision():
    """Enable mixed precision training for better performance"""
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

def configure_gpu_memory():
    """Configure GPU memory growth to prevent OOM errors"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
            # Set memory limit if specified
            if 'TF_MEMORY_LIMIT' in os.environ:
                limit = int(os.environ['TF_MEMORY_LIMIT'])
                for gpu in gpus:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=limit)]
                    )
        except RuntimeError as e:
            print(f"Error configuring GPU memory: {e}")
    else:
        print("No GPU devices found. Using CPU only.")

def set_tf_config():
    """Set TensorFlow configuration for optimal performance"""
    # Enable XLA optimization
    tf.config.optimizer.set_jit(True)
    
    # Set number of threads based on available CPU cores
    num_cores = psutil.cpu_count(logical=False)  # Physical cores only
    num_threads = max(1, num_cores // 2)  # Use half of available cores
    
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    
    # Enable operation fusion and auto mixed precision
    os.environ['TF_FP16_CONV_USE_FP32_COMPUTE'] = '0'
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    
    # Set memory growth and allocation options
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = str(num_threads)
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
    
def configure_performance():
    """Configure all performance optimizations"""
    enable_mixed_precision()
    configure_gpu_memory()
    set_tf_config()