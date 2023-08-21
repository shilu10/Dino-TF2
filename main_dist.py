import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import tqdm, datetime, os, sys 



def train_step(train_batch, teacher, student, epoch, dino_loss, optimizer, compute_loss):
    with tf.GradientTape() as tape:
        teacher_output = teacher(train_batch[:2])  # only the 2 global views pass through the teacher
        student_output = student(train_batch)
        loss = compute_loss(student_output, teacher_output, epoch)
    params = student.trainable_variables
    grads = tape.gradient(loss, params)
    optimizer.apply_gradients(zip(grads, params))
    
    return loss


@tf.function
def distributed_train_step(train_batch, teacher, student, epoch, dino_loss, optimizer, compute_loss, strategy):
    per_replica_losses = strategy.run(train_step,
                                    args=(train_batch, teacher, student, epoch, dino_loss, optimizer, compute_loss, ))

    return strategy.reduce(tf.distribute.ReduceOp.SUM,
                         per_replica_losses,
                         axis=None)


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
    data_path: str = None
  ):

    # distributed strategy
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # data_loader
    data_builder = tfds.folder_dataset.ImageFolder(data_path)
    dataset = data_builder.as_dataset(split='train',
                               shuffle_files=True,
                               batch_size=batch_size)

    multires_dataset = get_multires_dataset(dataset=dataset,
                          size_crops=[224, 96], # default values
                          num_crops=[2, local_crops_number],
                          min_scale = [global_crops_scale[0], local_crops_scale[0]],
                          max_scale = [global_crops_scale[1], local_crops_scale[1]],
                          batch_size=batch_size)

    dataloader = tf.data.Dataset.zip(multires_dataset)
    # distributed trainloader
    dist_dataloader = strategy.experimental_distribute_dataset(dataloader)

    print(f'len of dataloader: {len(dataloader)}')

    # distributed tf model
    with strategy.scope():
        student = ViTClassifier(config)
        teacher = ViTClassifier(config)

        embed_dim = student.config.projection_dim

        student = MultiCropWrapper(backbone = student,
                              head = DinoHead(embed_dim,
                                              out_dim,
                                              use_bn_in_head,
                                              norm_last_layer))

        teacher = MultiCropWrapper(backbone = teacher,
                              head = DinoHead(embed_dim,
                                              out_dim,
                                              use_bn_in_head))

    # distributed loss
    with strategy.scope():
        # Set reduction to `NONE` so you can do the reduction afterwards and divide by
        # global batch size.
        dino_loss = DinoLoss(
                out_dim,
                local_crops_number+2,
                warmup_teacher_temp,
                teacher_temp,
                warmup_teacher_temp_epochs,
                epochs,
            )

        def compute_loss(labels,
                     predictions,
                     epoch,
                     model_losses):

            per_example_loss = dino_loss(teacher_out=labels,
                                   student_out=predictions,
                                   epoch=epoch)

            loss = tf.nn.compute_average_loss(per_example_loss,
                                        global_batch_size=batch_size)
            if model_losses:
                loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
            
            return loss

    # distributed schedulers
    with strategy.scope():
        # ============ init schedulers ... ============
        lr_schedule = cosine_scheduler(
              lr * batch_size / 256.,  # linear scaling rule
              min_lr,
              epochs, len(dataloader),
              warmup_epochs=warmup_epochs,
          )

        wd_schedule = cosine_scheduler(
              weight_decay,
              weight_decay_end,
              epochs, len(dataloader),
          )

        # momentum parameter is increased to 1. during training with a cosine schedule
        momentum_schedule = cosine_scheduler(momentum_teacher, 1,
                                        epochs, len(dataloader))

    # metric logger
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    loss_logger = tf.keras.metrics.Mean(name='loss_logger')
    train_log_dir = 'dino/logs/gradient_tape/' + current_time + '/train/'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    scheduler_log_dir = 'dino/logs/gradient_tape/' + current_time + '/scheduler/'
    scheduler_summary_writer = tf.summary.create_file_writer(scheduler_log_dir)

    # distributed optimizer
    with strategy.scope():
        if optimizer == "adamw":
            optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)  # to use with ViTs

        elif optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=0,
                                          momentum=0.9,
                                          weight_decay=weight_decay)  # lr is set by scheduler
            
    # checkpoint manager and checkpoint
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                     optimizer=optimizer, 
                                     teacher=teacher, 
                                     student=student)
    
    ckpt_manager = tf.train.CheckpointManager(checkpoint=checkpoint, 
                                             directory="/tmp/model", 
                                             max_to_keep=5)
    
    
    # restore latest checkpoint if an
    checkpoint.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Loaded from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initialize from scratch.")

    for epoch in range(epochs):
        epoch_loss = 0
        print(f'epoch: {epoch}')

        for indx, train_batch in enumerate(tqdm.tqdm(dist_dataloader), total=len(dataloader)):
            indx = len(dataloader) * epoch + indx  # global training iteration

            # update lr and weight decay values
            optimizer.learning_rate = lr_schedule[indx]
            optimizer.weight_decay = wd_schedule[indx]
            
            loss = distributed_train_step(
              train_batch = train_batch,
              teacher = teacher,
              student = student,
              epoch = epoch,
              dino_loss = dino_loss,
              optimizer = optimizer,
              compute_loss = compute_loss,
              strategy = strategy
            )

            loss_logger.update_states(loss)

            # update teacher model
            m = momentum_schedule[indx]  # momentum paramete
            teacher_weights = teacher.get_weights()
            student_weights = student.get_weights()
            
            for weight_indx in range(len(student_weights)):
                teacher_weights[weight_indx] = (teacher_weights[weight_indx] * m) + (student_weights[weight_indx] * (1 - m))
                
            teacher.set_weights(teacher_weights)
            
            with scheduler_summary_writer.as_default():
                tf.summary.scalar('lr', lr_schedule[indx], step=indx)
                tf.summary.scalar('weight_decay', wd_schedule[indx], step=indx)
                
        checkpoint.step.assing_add(1)
        if int(checkpoint.step) % 10 == 0:
            save_path = ckpt_manager.save()
            print("Saved checkpoint at step {}: {}".format(epoch, save_path))
        
        epoch_loss = loss_logger.result()
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', epoch_loss, step=epoch)

        print(f'epoch; {epoch}: {loss_logger.result()}')
        loss_logger.reset_states()