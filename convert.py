from tensorflow import keras 
import tensorflow as tf 
import numpy as np 
import timm, os, sys 
import yaml
from vision_transformer import ViTClassifier, vit_tiny, vit_small, vit_base, TFViTAttention
from utils import * 
import argparse
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



def get_args_parser():
    parser = argparse.ArgumentParser('dino-porting', add_help=False)

    parser.add_argument('--arch', default='vit_small_patch8_224.dino', type=str,
        choices=['vit_small_patch8_224.dino', 'vit_small_patch16_224.dino', 
                            'vit_base_patch16_224.dino', 'vit_base_patch8_224.dino'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")

    parser.add_argument('--include_top', default=False, type=bool, 
        help="""used for including the trained linear layers""")

    parser.add_argument('--model_savepath', default="models/", type=str, 
        help='Used for saving the ported tensorflow models')

    parser.add_argument('--store_model', default=True, type=bool, 
        help="""If True, then the following function, will store the 
        tensorflow model in form of stored_model, if false, then it 
        stores the model_weights""")
    
    parser.add_argument('--patch_size', default=16, type=int, help="specify the patch_size.")

    return parser


def port(args):
    model_type = args.arch
    include_top = args.include_top
    model_savepath = args.model_savepath
    num_classes = 0 if not include_top else 1000

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if not "base" in model_type and include_top:
        raise NotImplementedError("Given Combination is not available.")

    print("Instantiating PyTorch model...")
    pt_model = timm.create_model(
        model_name=model_type, 
        num_classes=1000, 
        pretrained=True
    )

    pt_model.eval()

    print("Instantiating TF model...")
    if "tiny" in model_type:
        tf_model, config = vit_tiny(return_config=True, 
                                    patch_size=args.patch_size, 
                                    num_classes=num_classes)
    
    elif "base" in model_type:
        tf_model, config = vit_base(return_config=True, 
                                    patch_size=args.patch_size, 
                                    num_classes=num_classes)

    elif "small" in model_type:
        tf_model, config = vit_small(return_config=True, 
                                    patch_size=args.patch_size, 
                                    num_classes=num_classes)
    
    else:
        raise NotImplementedError('given model_type is not implemented')

    image_dim = 224
    dummy_inputs = tf.ones((2, image_dim, image_dim, 3))
    _ = tf_model(dummy_inputs)[0]

    # Load the PT params.
    pt_model_dict = pt_model.state_dict()
    pt_model_dict = {k: pt_model_dict[k].numpy() for k in pt_model_dict}

    if include_top:
        urls = {
            "vit_base_patch8_224.dino": 'wget https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth -q',
            "vit_base_patch16_224.dino": 'wget https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth -q'
        }
        # download weights (olinear weights)
        cmd = urls[model_type]
        os.system(cmd)

        if "vit_base_patch8_224.dino" in model_type:
            linear_url = 'dino_vitbase8_linearweights.pth'
        else:
            linear_url = 'dino_vitbase16_linearweights.pth'

        linear_state_dict = torch.load(linear_url, map_location=torch.device("cpu")).get('state_dict')
        
        pt_model_dict['head.weight'] = np.array(linear_state_dict['module.linear.weight'])
        pt_model_dict['head.bias'] = np.array(linear_state_dict['module.linear.bias'])

    print("Beginning parameter porting process...")

    # classification head
    if include_top:
        tf_model.layers[-1] = modify_tf_block(
            tf_model.layers[-1],
            pt_model_dict["head.weight"],
            pt_model_dict["head.bias"],
        )

    # Projection layers.
    tf_model.layers[0].patch_embeddings.projection = modify_tf_block(
        tf_model.layers[0].patch_embeddings.projection,
        pt_model_dict["patch_embed.proj.weight"],
        pt_model_dict["patch_embed.proj.bias"],
    )

    # Positional embedding.
    tf_model.layers[0].position_embeddings.assign(
        tf.Variable(pt_model_dict["pos_embed"])
    )

    # CLS and (optional) Distillation tokens.
    tf_model.layers[0].cls_token.assign(tf.Variable(pt_model_dict["cls_token"]))

    # Layer norm layers.
    ln_idx = -1 if not include_top else -2 
    tf_model.layers[ln_idx] = modify_tf_block(
        tf_model.layers[ln_idx],
        pt_model_dict["norm.weight"],
        pt_model_dict["norm.bias"],
    )


    # Transformer blocks.
    idx = 0

    for outer_layer in tf_model.layers:
        if (
            isinstance(outer_layer, tf.keras.Model)
            and outer_layer.name != "projection"
        ):
            tf_block = tf_model.get_layer(outer_layer.name)
            pt_block_name = f"blocks.{idx}"

            # LayerNorm layers.
            layer_norm_idx = 1
            for layer in tf_block.layers:
                if isinstance(layer, tf.keras.layers.LayerNormalization):
                    layer_norm_pt_prefix = (
                        f"{pt_block_name}.norm{layer_norm_idx}"
                    )
                    layer.gamma.assign(
                        tf.Variable(
                            pt_model_dict[f"{layer_norm_pt_prefix}.weight"]
                        )
                    )
                    layer.beta.assign(
                        tf.Variable(
                            pt_model_dict[f"{layer_norm_pt_prefix}.bias"]
                        )
                    )
                    layer_norm_idx += 1

            # FFN layers.
            ffn_layer_idx = 1
            for layer in tf_block.layers:
                if isinstance(layer, tf.keras.layers.Dense):
                    dense_layer_pt_prefix = (
                        f"{pt_block_name}.mlp.fc{ffn_layer_idx}"
                    )
                    layer = modify_tf_block(
                        layer,
                        pt_model_dict[f"{dense_layer_pt_prefix}.weight"],
                        pt_model_dict[f"{dense_layer_pt_prefix}.bias"],
                    )
                    ffn_layer_idx += 1


            # Attention layer.
            for layer in tf_block.layers:
                (q_w, k_w, v_w), (q_b, k_b, v_b) = get_tf_qkv(
                    f"{pt_block_name}.attn",
                    pt_model_dict,
                    config,
                )

                if isinstance(layer, TFViTAttention):
                    # Key
                    layer.self_attention.key = modify_tf_block(
                        layer.self_attention.key,
                        k_w,
                        k_b,
                        is_attn=True,
                    )
                    # Query
                    layer.self_attention.query = modify_tf_block(
                        layer.self_attention.query,
                        q_w,
                        q_b,
                        is_attn=True,
                    )
                    # Value
                    layer.self_attention.value = modify_tf_block(
                        layer.self_attention.value,
                        v_w,
                        v_b,
                        is_attn=True,
                    )
                    # Final dense projection
                    layer.dense_output.dense = modify_tf_block(
                        layer.dense_output.dense,
                        pt_model_dict[f"{pt_block_name}.attn.proj.weight"],
                        pt_model_dict[f"{pt_block_name}.attn.proj.bias"],
                    )

            for layer in tf_block.layers:

              if isinstance(layer, tf.keras.Sequential):
                d_indx = 1
                for indx, inner_layer in enumerate(layer.layers):
                  if len(layer.layers) >= 2 and isinstance(inner_layer, tf.keras.layers.Dense):
                    inner_layer = modify_tf_block(
                            inner_layer,
                            pt_model_dict[f"{pt_block_name}.mlp.fc{d_indx}.weight"],
                            pt_model_dict[f"{pt_block_name}.mlp.fc{d_indx}.bias"],
                        )
                    d_indx += 1

            idx += 1

    print("Porting successful, serializing TensorFlow model...")

    save_path = os.path.join(model_savepath, model_type)

    if args.store_model:
        tf_model.save(save_path)
    else:
        tf_model.save_weights(save_path)

    print(f"TensorFlow model serialized at: {save_path}...")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    port(args)