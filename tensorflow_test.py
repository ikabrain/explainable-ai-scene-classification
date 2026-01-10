import tensorflow as tf

# 1. Check for GPU availability
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print(f"✅ SUCCESS: TensorFlow found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"   - {gpu}")

    # 2. explicit check if it can compute on GPU
    try:
        with tf.device("/GPU:0"):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print("\n✅ GPU Computation Test Passed!")
    except RuntimeError as e:
        print(f"\n❌ GPU Found but computation failed: {e}")

else:
    print("❌ FAILURE: TensorFlow is running on CPU only.")
    print(
        "   - Please ensure that your system has a compatible GPU and the necessary drivers installed."
    )
