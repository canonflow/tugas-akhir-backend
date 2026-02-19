import tensorflow as tf
import numpy as np

print("\n============================ SETTING GPU IF EXISTS ============================")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Prevent TensorFlow from reserving all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class ReduceMeanLayer(tf.keras.layers.Layer):
    def __init__(self, axis=-1, keepdims=True, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims
    
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=self.axis, keepdims=self.keepdims)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "keepdims": self.keepdims,
        })
        return config


class ReduceMaxLayer(tf.keras.layers.Layer):
    def __init__(self, axis=-1, keepdims=True, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims
    
    def call(self, inputs):
        return tf.reduce_max(inputs, axis=self.axis, keepdims=self.keepdims)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "keepdims": self.keepdims,
        })
        return config


class L2NormalizationLayer(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
    
    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=self.axis)
    
    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


class CosineSimilarityLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        feat_ref, feat_sketch = inputs
        # Features are already L2 normalized, so dot product = cosine similarity
        return tf.reduce_sum(feat_ref * feat_sketch, axis=1, keepdims=True)
    
    def get_config(self):
        return super().get_config()

def cosine_similarity(vectors):
    x, y = vectors
    x = tf.nn.l2_normalize(x, axis=1)
    y = tf.nn.l2_normalize(y, axis=1)
    return tf.reduce_sum(x * y, axis=1, keepdims=True)  # hasil shape (batch, 1)


    
def pearson_correlation_metric(y_true, y_pred):
        """
        Calculate Pearson correlation coefficient as a metric
        """
        # Flatten the tensors to ensure proper calculation
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])

        # Calculate means
        mean_true = tf.reduce_mean(y_true_flat)
        mean_pred = tf.reduce_mean(y_pred_flat)

        # Center the variables
        centered_true = y_true_flat - mean_true
        centered_pred = y_pred_flat - mean_pred

        # Calculate numerator and denominators
        numerator = tf.reduce_sum(centered_true * centered_pred)

        sum_sq_true = tf.reduce_sum(tf.square(centered_true))
        sum_sq_pred = tf.reduce_sum(tf.square(centered_pred))

        denominator = tf.sqrt(sum_sq_true * sum_sq_pred)

        # Calculate correlation with epsilon for numerical stability
        correlation = numerator / (denominator + tf.keras.backend.epsilon())

        # Clip to ensure it's in [-1, 1] range due to potential floating point errors
        correlation = tf.clip_by_value(correlation, -1.0, 1.0)

        return correlation

def build_efficientnetv2m_subnetwork(input_shape=256):
    base = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=(input_shape, input_shape, 3)
    )

    base.trainable = False

    x = base.output

    # Spatial Attention mechanism
    avg_pool = ReduceMeanLayer(axis=-1, keepdims=True)(x)
    max_pool = ReduceMaxLayer(axis=-1, keepdims=True)(x)

    concat = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
    attention = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=7,
        strides=1,
        padding='same',
        activation='sigmoid',
    )(concat)

    x = tf.keras.layers.Multiply()([x, attention])

    # Hybrid Pooling
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling2D()(x)
    hybrid_pool = tf.keras.layers.Concatenate()([avg_pool, max_pool])

    # TODO: [UPDATE - After _4_with_full_cbam] Dense layers - Corrected order: Dense → BN → Activation → Dropout
    fc = tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(hybrid_pool)
    fc = tf.keras.layers.BatchNormalization()(fc)
    fc = tf.keras.layers.Activation('relu')(fc)
    fc = tf.keras.layers.Dropout(0.5)(fc)

    fc = tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(fc)
    fc = tf.keras.layers.BatchNormalization()(fc)
    fc = tf.keras.layers.Activation('relu')(fc)
    fc = tf.keras.layers.Dropout(0.4)(fc)

    fc = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(fc)
    fc = tf.keras.layers.BatchNormalization()(fc)
    fc = tf.keras.layers.Activation('relu')(fc)
    fc = tf.keras.layers.Dropout(0.3)(fc)

    fc = L2NormalizationLayer(axis=1)(fc)

    return tf.keras.Model(base.input, fc, name="feature_extractor")

def build_convnext_subnetwork(input_shape=224):
    base = tf.keras.applications.ConvNeXtSmall(
        include_top=False,
        weights='imagenet',
        input_shape=(input_shape, input_shape, 3)
    )

    base.trainable = False

    x = base.output

    # Spatial Attention mechanism
    avg_pool = ReduceMeanLayer(axis=-1, keepdims=True)(x)
    max_pool = ReduceMaxLayer(axis=-1, keepdims=True)(x)

    concat = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
    attention = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=7,
        strides=1,
        padding='same',
        activation='sigmoid',
    )(concat)

    x = tf.keras.layers.Multiply()([x, attention])

    # Hybrid Pooling
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling2D()(x)
    hybrid_pool = tf.keras.layers.Concatenate()([avg_pool, max_pool])

    # TODO: [UPDATE - After _4_with_full_cbam] Dense layers - Corrected order: Dense → BN → Activation → Dropout
    fc = tf.keras.layers.Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(hybrid_pool)
    fc = tf.keras.layers.BatchNormalization()(fc)
    fc = tf.keras.layers.Activation('relu')(fc)
    fc = tf.keras.layers.Dropout(0.5)(fc)

    fc = tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(fc)
    fc = tf.keras.layers.BatchNormalization()(fc)
    fc = tf.keras.layers.Activation('relu')(fc)
    fc = tf.keras.layers.Dropout(0.4)(fc)

    fc = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(fc)
    fc = tf.keras.layers.BatchNormalization()(fc)
    fc = tf.keras.layers.Activation('relu')(fc)
    fc = tf.keras.layers.Dropout(0.3)(fc)

    fc = L2NormalizationLayer(axis=1)(fc)

    return tf.keras.Model(base.input, fc, name="feature_extractor")

def build_siamese(input_shape):
    print("--- DEFINE INPUT LAYERS")
    input_ref = tf.keras.Input(shape=(input_shape, input_shape, 3), name="reference")
    input_sketch = tf.keras.Input(shape=(input_shape, input_shape, 3), name="sketch")

    print("--- DEFINE SUB-NETWORK")
    subnet = build_convnext_subnetwork()
    feat_ref = subnet(input_ref)
    feat_sketch = subnet(input_sketch)

    output = CosineSimilarityLayer(name='cosine_similarity')([feat_ref, feat_sketch])
    output = tf.keras.layers.Lambda(lambda x: x * 100)(output)

    siamese_model = tf.keras.Model([input_ref, input_sketch], output)

    siamese_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mse', tf.keras.metrics.R2Score(), pearson_correlation_metric]
    )

    return siamese_model

def build_callbacks():
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_mse',
        patience=30,
        restore_best_weights=True,
        mode='min'
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_r2_score',
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )

    return [early_stopping, reduce_lr]

class SiameseModel:
    def __init__(self, ref_image, sketch_image):
        self.ref_image = np.expand_dims(ref_image, axis=0) if ref_image.ndim == 3 else ref_image
        self.sketch_image = np.expand_dims(sketch_image, axis=0) if sketch_image.ndim == 3 else sketch_image
        self.model = self.build_model()

    def calculate_similarity(self):
        prediction = self.model.predict([self.ref_image, self.sketch_image], verbose=1)
        score = prediction.squeeze()
        return score

    def build_model(self):
        return tf.keras.models.load_model(
            "model/siamese_model.keras",
            custom_objects={
                'ReduceMeanLayer': ReduceMeanLayer,
                'ReduceMaxLayer': ReduceMaxLayer,
                'L2NormalizationLayer': L2NormalizationLayer,
                'CosineSimilarityLayer': CosineSimilarityLayer,
                'pearson_correlation_metric': self.pearson_correlation_metric
            },
            safe_mode=False
        )

    def pearson_correlation_metric(y_true, y_pred):
        """
        Calculate Pearson correlation coefficient as a metric
        """
        # Flatten the tensors to ensure proper calculation
        y_true_flat = tf.reshape(y_true, [-1])
        y_pred_flat = tf.reshape(y_pred, [-1])

        # Calculate means
        mean_true = tf.reduce_mean(y_true_flat)
        mean_pred = tf.reduce_mean(y_pred_flat)

        # Center the variables
        centered_true = y_true_flat - mean_true
        centered_pred = y_pred_flat - mean_pred

        # Calculate numerator and denominators
        numerator = tf.reduce_sum(centered_true * centered_pred)

        sum_sq_true = tf.reduce_sum(tf.square(centered_true))
        sum_sq_pred = tf.reduce_sum(tf.square(centered_pred))

        denominator = tf.sqrt(sum_sq_true * sum_sq_pred)

        # Calculate correlation with epsilon for numerical stability
        correlation = numerator / (denominator + tf.keras.backend.epsilon())

        # Clip to ensure it's in [-1, 1] range due to potential floating point errors
        correlation = tf.clip_by_value(correlation, -1.0, 1.0)

        return correlation

