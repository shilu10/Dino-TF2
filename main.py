def train_dino(
    arch: str = "vit_base",
    patch_size: int = 16,
    out_dim: int = 65_536,
    norm_last_layer: bool = True,
    momentum_teacher: float = 0.996,
    use_bn_in_head: bool = False,
    lr: float = 0.0005,
    batch_size: int = 64,
    weight_decay: float = 0.04,
    weight_decay_end: float = 0.4,
    epochs: int = 100,
    warmup_epochs: int = 10,
    min_lr: float = 1e-6,
    optimizer: str = "adamw",
    global_crops_scale: Tuple[float, float] = (0.4, 1.0),
    local_crops_number: int = 5,
    local_crops_scale: Tuple[float, float] = (0.05, 0.4),
    warmup_teacher_temp: float = 0.04,
    teacher_temp: float = 0.04,
    warmup_teacher_temp_epochs: int = 0,
    trainloader=None
  ):

  student = ViTClassifier(config, False)
  teacher = ViTClassifier(config, False)

  embed_dim = student.config.projection_dim

  student = MultiCropWrapper(student, DinoHead(embed_dim, out_dim, use_bn_in_head, norm_last_layer))
  teacher = MultiCropWrapper(teacher, DinoHead(embed_dim, out_dim, use_bn_in_head))

  dino_loss = DinoLoss(
      out_dim,
      local_crops_number+2,
      warmup_teacher_temp,
      teacher_temp,
      warmup_teacher_temp_epochs,
      epochs,
  )

  # ============ init schedulers ... ============
  lr_schedule = cosine_scheduler(
        lr * batch_size / 256.,  # linear scaling rule
        min_lr,
        epochs, len(trainloader),
        warmup_epochs=warmup_epochs,
    )
  wd_schedule = cosine_scheduler(
        weight_decay,
        weight_decay_end,
        epochs, len(trainloader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
  momentum_schedule = cosine_scheduler(momentum_teacher, 1,
                                               epochs, len(trainloader))

  opt = tf.keras.optimizers.SGD(learning_rate=lr)

  for epoch in range(epochs):
    epoch_loss = 0
    print(f'epoch: {epoch}')
    for indx, train_batch in enumerate(tqdm.tqdm(trainloader)):
      indx = len(trainloader) * epoch + indx  # global training iteration

      # update lr and weight decay values
      opt.learning_rate = lr_schedule[indx]
      opt.weight_decay = wd_schedule[indx]

      with tf.GradientTape() as tape:
        teacher_output = teacher(train_batch[:2])  # only the 2 global views pass through the teacher
        student_output = student(train_batch)
        loss = dino_loss(student_output, teacher_output, epoch)

      params = student.trainable_variables
      grads = tape.gradient(loss, params)
      opt.apply_gradients(zip(grads, params))

      epoch_loss += loss

      # update teacher model
      m = momentum_schedule[indx]  # momentum parameter
      teacher_weights = teacher.get_weights()
      student_weights = student.get_weights()
      for weight_indx in range(len(student_weights)):
        teacher_weights[weight_indx] = (teacher_weights[weight_indx] * m) + (student_weights[weight_indx] * (1 - m))

      teacher.set_weights(teacher_weights)

    print(f'epoch; {epoch}: {epoch_loss}')
