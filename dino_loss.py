import tensorflow as tf 
from tensorflow import keras 
import numpy as np 


class DinoLoss(tf.keras.losses.Loss):
  def __init__(self,
               out_dim: int = 65536,
               ncrops: int = 7,
               warmup_teacher_temp : float =0.004,
               teacher_temp: float = 0.04,
               warmup_teacher_temp_epochs: int = 0,
               nepochs: int = 100,
               student_temp: float = 0.1,
               center_momentum: float = 0.9,
               **kwargs
            )-> tf.keras.losses.Loss:

    super(DinoLoss, self).__init__(**kwargs)
    self.student_temp = student_temp
    self.center_momentum = center_momentum
    self.ncrops = ncrops

    # we apply a warm up for the teacher temperature because
    # a too high temperature makes the training instable at the beginning
    self.teacher_temp_schedule = np.concatenate((
          np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
          np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
      ))
    self.center = tf.zeros_like(out_dim, dtype=tf.float32)

  def __call__(self,
               student_output: tf.Tensor,
               teacher_output: tf.Tensor,
               epoch: int) -> tf.Tensor:
    """
      Cross-entropy between softmax outputs of the teacher and student networks.
    """
    #super.__call__()
    student_out = student_output / self.student_temp
    student_out = tf.split(student_out, self.ncrops)

    # teacher centering and sharpening
    temp = self.teacher_temp_schedule[epoch]
    teacher_out = tf.stop_gradient(tf.nn.softmax((teacher_output - self.center) / temp, axis=-1))
    teacher_out = tf.split(teacher_out, 2)


    total_loss = 0
    n_loss_terms = 0
    for iq, q in enumerate(teacher_out):
      for v in range(len(student_out)):
        if v == iq:
          # we skip cases where student and teacher operate on the same view
          continue
        loss = tf.reduce_sum(
                    -q * tf.nn.log_softmax(student_out[v], axis=-1), axis=-1
                )

        # total_loss += tf.math.reduce_mean(loss) # use for one-gpu training
        total_loss += loss # for multi-gpu
        n_loss_terms += 1

    total_loss /= n_loss_terms
    self.update_center(teacher_output)
    return total_loss

  def update_center(self,
                    teacher_output: tf.Tensor)-> None:
    """
      Update center used for teacher output.
    """
    batch_center = tf.math.reduce_sum(teacher_output, axis=0)
    batch_center = batch_center / (len(teacher_output))
    self.center = tf.stop_gradient(
            self.center * self.center_momentum
            + batch_center * (1 - self.center_momentum)
        )

  def get_config(self):
    config = super().get_config()
    config.update(
        {
            'student_temp': self.student_temp,
           # 'center_momentum': self.center_momentum,
            'ncrops': self.ncrops
        }
      )

    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)


    
    