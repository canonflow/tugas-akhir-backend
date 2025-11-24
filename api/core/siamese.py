import tensorflow as tf
import numpy as np

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

