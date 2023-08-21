import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import tqdm, datetime, os, sys 
from typing import * 
import argparse
from vision_transformer import *
from data_augmentaton import get_multires_dataset
from dino_loss import DinoLoss
from dino_head import DinoHead
from multicrop_wrapper import MultiCropWrapper
from utils import * 
import tensorflow_datasets as tfds


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")

    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")

    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")

    parser.add_argument('--norm_last_layer', default=True, type=bool,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")

    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")

    parser.add_argument('--use_bn_in_head', default=False, type=bool,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")

    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")

    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")

    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
 
    parser.add_argument('--batch_size', default=64, type=int,
        help='batch_size : number of distinct images loaded on  GPU.')

    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')

    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")

    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")

    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")

    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
        
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
        
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)

    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')

    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')

    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')


    return parser


def train_step(train_batch: tf.Tensor, 
               teacher: tf.keras.Model, 
               student: tf.keras.Model, 
               epoch: int, 
               dino_loss: tf.keras.losses.Loss, 
               optimizer: tf.keras.optimizers.Optimizer, 
               compute_loss: Callable):
    
    with tf.GradientTape() as tape:
        teacher_output = teacher(train_batch[:2])  # only the 2 global views pass through the teacher
        student_output = student(train_batch)
        loss = compute_loss(student_output, teacher_output, epoch)
    params = student.trainable_variables
    grads = tape.gradient(loss, params)
    optimizer.apply_gradients(zip(grads, params))
    
    return loss


@tf.function
def distributed_train_step(train_batch: tf.Tensor, 
                           teacher: tf.keras.Model, 
                           student: tf.keras.Model, 
                           epoch: int, 
                           dino_loss: tf.keras.losses.Loss, 
                           optimizer: tf.keras.optimizers.Optimizer, 
                           compute_loss: Callable,
                           strategy):
    
    per_replica_losses = strategy.run(train_step,
                                    args=(train_batch, teacher, student, epoch, dino_loss, optimizer, compute_loss, ))

    return strategy.reduce(tf.distribute.ReduceOp.SUM,
                         per_replica_losses,
                         axis=None)


def train_dino(args):

    # distributed strategy
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # data_loader
    data_builder = tfds.folder_dataset.ImageFolder(args.data_path)
    dataset = data_builder.as_dataset(split='train',
                               shuffle_files=True,
                               batch_size=args.batch_size)

    multires_dataset = get_multires_dataset(dataset=dataset,
                          size_crops=[224, 96], # default values
                          num_crops=[2, args.local_crops_number],
                          min_scale = [args.global_crops_scale[0], args.local_crops_scale[0]],
                          max_scale = [args.global_crops_scale[1], args.local_crops_scale[1]],
                          batch_size=args.batch_size)

    dataloader = tf.data.Dataset.zip(multires_dataset)
    # distributed trainloader
    dist_dataloader = strategy.experimental_distribute_dataset(dataloader)

    print(f'len of dataloader: {len(dataloader)}')

    if args.arch == 'vit_small':
        student = vit_small(patch_size=args.patch_size)
        teacher = vit_small(patch_size=args.patch_size)

    elif args.arch == 'vit_base':
        student = vit_base(patch_size=args.patch_size)
        teacher = vit_base(patch_size=args.patch_size)

    elif args.arch == 'vit_tiny':
        student = vit_tiny(patch_size=args.patch_size)
        teacher = vit_tiny(patch_size=args.patch_size)

    else:
        raise NotImplementedError("Given Model variant is not supported")

    embed_dim = student.config.projection_dim

    # distributed tf model
    with strategy.scope():
        student = MultiCropWrapper(backbone = student,
                              head = DinoHead(embed_dim,
                                              args.out_dim,
                                              args.use_bn_in_head,
                                              args.norm_last_layer))

        teacher = MultiCropWrapper(backbone = teacher,
                              head = DinoHead(embed_dim,
                                              args.out_dim,
                                              args.use_bn_in_head))

    # distributed loss
    with strategy.scope():
        # Set reduction to `NONE` so you can do the reduction afterwards and divide by
        # global batch size.
        dino_loss = DinoLoss(
                args.out_dim,
                args.local_crops_number+2,
                args.warmup_teacher_temp,
                args.teacher_temp,
                args.warmup_teacher_temp_epochs,
                args.epochs,
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
              args.lr * args.batch_size / 256.,  # linear scaling rule
              args.min_lr,
              args.epochs, len(dataloader),
              warmup_epochs=args.warmup_epochs,
          )

        wd_schedule = cosine_scheduler(
              args.weight_decay,
              args.weight_decay_end,
              args.epochs, len(dataloader),
          )

        # momentum parameter is increased to 1. during training with a cosine schedule
        momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
                                        args.epochs, len(dataloader))

    # metric logger
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    loss_logger = tf.keras.metrics.Mean(name='loss_logger')
    train_log_dir = args.output_dir + '/logs/' + current_time + '/train/'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    scheduler_log_dir = args.output_dir + '/logs/' + current_time + '/scheduler/'
    scheduler_summary_writer = tf.summary.create_file_writer(scheduler_log_dir)


    # distributed optimizer
    with strategy.scope():
        if args.optimizer == "adamw":
            optimizer = tf.keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)  # to use with ViTs

        elif args.optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=0,
                                          momentum=0.9,
                                          weight_decay=args.weight_decay)  # lr is set by scheduler
            
    # checkpoint manager and checkpoint
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                     optimizer=optimizer, 
                                     teacher=teacher, 
                                     student=student)
    
    ckpt_manager = tf.train.CheckpointManager(checkpoint=checkpoint, 
                                             directory=args.output_dir + "/dino_model/", 
                                             max_to_keep=5)
    
    
    # restore latest checkpoint if an
    checkpoint.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Loaded from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initialize from scratch.")

    for epoch in range(args.epochs):
        epoch_loss = 0
        print(f'epoch: {epoch}')

        for indx, train_batch in enumerate(tqdm.tqdm(dist_dataloader, total=len(dataloader))):
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
        if int(checkpoint.step) % args.saveckp_freq == 0:
            save_path = ckpt_manager.save()
            print("Saved checkpoint at step {}: {}".format(epoch, save_path))
        
        epoch_loss = loss_logger.result()
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', epoch_loss, step=epoch)

        print(f'epoch; {epoch}: {loss_logger.result()}')
        loss_logger.reset_states()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    train_dino(args)