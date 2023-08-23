import os 
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 
from tensorflow import keras 

from tqdm import tqdm
from PIL import Image

from vision_transformer import * 


FOURCC = {
    "mp4": cv2.VideoWriter_fourcc(*"MP4V"),
    "avi": cv2.VideoWriter_fourcc(*"XVID"),
}


class VideoGeneratorTF:
    def __init__(self, args):
        self.args = args

        # For DeiT, DINO this should be unchanged. For the original ViT-B16 models,
        # input images should be scaled to [-1, 1] range.
        self.norm_layer = keras.layers.Normalization(
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            variance=[(0.229 * 255) ** 2, (0.224 * 255) ** 2, (0.225 * 255) ** 2],
        )
        
        if not self.args.video_only:
            self.model = self.__load_model()

    def run(self):
        if self.args.input_path is None:
            print(f"Provided input path {self.args.input_path} is non valid.")
            sys.exit(1)
        else:
            if self.args.video_only:
                self._generate_video_from_images(
                    self.args.input_path, self.args.output_path
                )
                
            else:
                # If input path exists
                if os.path.exists(self.args.input_path):
                    # If input is a video file
                    if os.path.isfile(self.args.input_path):
                        frames_folder = os.path.join(self.args.output_path, f"{self.args.output_filename}_frames-tf")
                        attention_folder = os.path.join(
                            self.args.output_path, f"{self.args.output_filename}_attention-tf"
                        )

                        os.makedirs(frames_folder, exist_ok=True)
                        os.makedirs(attention_folder, exist_ok=True)

                        self._extract_frames_from_video(
                            self.args.input_path, frames_folder
                        )

                        self._inference(
                            frames_folder,
                            attention_folder,
                        )

                        self._generate_video_from_images(
                            attention_folder, self.args.output_path
                        )
                        
                        filepath = f'{self.args.output_path}/{self.args.output_filename}.{self.args.video_format}'
                        print(f'Video Serialized at, {filepath}')

                    # If input is a folder of already extracted frames
                    if os.path.isdir(self.args.input_path):
                        attention_folder = os.path.join(
                            self.args.output_path, "attention-tf"
                        )

                        os.makedirs(attention_folder, exist_ok=True)

                        self._inference(self.args.input_path, attention_folder)

                        self._generate_video_from_images(
                            attention_folder, self.args.output_path
                        )
                        
                        filepath = f'{self.args.output_path}/{self.args.output_filename}.{self.args.video_format}'
                        print(f'Video Serialized at, {filepath}')

                # If input path doesn't exists
                else:
                    print(f"Provided input path {self.args.input_path} doesn't exists.")
                    sys.exit(1)

    def _extract_frames_from_video(self, inp: str, out: str):
        vidcap = cv2.VideoCapture(inp)
        self.args.fps = vidcap.get(cv2.CAP_PROP_FPS)

        print(f"Video: {inp} ({self.args.fps} fps)")
        print(f"Extracting frames to {out}")

        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(
                os.path.join(out, f"frame-{count:04}.jpg"),
                image,
            )
            success, image = vidcap.read()
            count += 1

    def _generate_video_from_images(self, inp: str, out: str):
        img_array = []
        attention_images_list = sorted(glob.glob(os.path.join(inp, "attn-*.jpg")))

        # Get size of the first image
        with open(attention_images_list[0], "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            size = (img.width, img.height)
            img_array.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

        print(f"Generating video {size} to {out}")

        for filename in tqdm(attention_images_list[1:]):
            with open(filename, "rb") as f:
                img = Image.open(f)
                img = img.convert("RGB")
                img_array.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

        out = cv2.VideoWriter(
            os.path.join(out, f"{self.args.output_filename}." + self.args.video_format),
            FOURCC[self.args.video_format],
            self.args.fps,
            size,
        )

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        print("Done")

    def _preprocess_image(self, image: Image, size: int):
        # Reference: https://www.tensorflow.org/lite/examples/style_transfer/overview
        image = np.array(image)
        image_resized = tf.expand_dims(image, 0)
        shape = tf.cast(tf.shape(image_resized)[1:-1], tf.float32)
        short_dim = min(shape)
        scale = size / short_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        image_resized = tf.image.resize(
            image_resized,
            new_shape,
        )
        return self.norm_layer(image_resized).numpy()

    def _inference(self, inp: str, out: str):
        print(f"Generating attention images to {out}")

        for img_path in tqdm(sorted(glob.glob(os.path.join(inp, "*.jpg")))):
            with open(img_path, "rb") as f:
                img = Image.open(f)
                img = img.convert("RGB")

            preprocessed_image = self._preprocess_image(img, self.args.resize)
            h, w = (
                preprocessed_image.shape[1]
                - preprocessed_image.shape[1] % self.args.patch_size,
                preprocessed_image.shape[2]
                - preprocessed_image.shape[2] % self.args.patch_size,
            )
            preprocessed_image = preprocessed_image[:, :h, :w, :]

            h_featmap = preprocessed_image.shape[1] // self.args.patch_size
            w_featmap = preprocessed_image.shape[2] // self.args.patch_size

            # Grab the attention scores from the final transformer block.
            attentions = self.model.get_last_selfattention(preprocessed_image, training=False)
            attentions = attentions.numpy()

            nh = attentions.shape[1]  # number of head

            # we keep only the output patch attention
            attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
            attentions = attentions.reshape(nh, h_featmap, w_featmap)
            attentions = attentions.transpose((1, 2, 0))

            # interpolate
            attentions = tf.image.resize(
                attentions,
                size=(
                    h_featmap * self.args.patch_size,
                    w_featmap * self.args.patch_size,
                ),
            )

            # save attentions heatmaps
            fname = os.path.join(out, "attn-" + os.path.basename(img_path))
            plt.imsave(
                fname=fname,
                arr=sum(
                    attentions[..., i] * 1 / attentions.shape[-1]
                    for i in range(attentions.shape[-1])
                ),
                cmap="inferno",
                format="jpg",)
            
    
    def __load_model(self):
        if args.use_pretrained_weights:
            if 'base' in args.arch:
                model = vit_base(num_classes=0, patch_size=args.patch_size)
            elif 'small' in args.arch:
                model = vit_small(num_classes=0, patch_size=args.patch_size)
            elif 'tiny' in args.arch:
                model = vit_tiny(num_classes=0, patch_size=args.patch_size)
            else:
                print('specified architecture is not presented')
                sys.exit(1)

            model.load_weights(args.pretrained_weights)
            
        else:
            tf.load_model(args.model_path, compile=False)
            
        return model


def get_arg_parser():
    parser = argparse.ArgumentParser("Generation self-attention video")
    parser.add_argument(
        "--arch",
        default="vit_base",
        type=str,
        choices=["vit_tiny", "vit_small", "vit_base"],
        help="Architecture (support only ViT atm).",
    )
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the self.model."
    )
    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="Path to pretrained weights to load.",
    )
    parser.add_argument(
        "--use_pretrained_weights",
        default=True,
        type=bool,
        help="""use_pretrained_weights=True, then pretrained_weights 
        path should be provided, else model_path should be provided""",
    )
    parser.add_argument(
        "--model_path",
        default="",
        type=str,
        help='path of the saved tensorflow model.',
    )
    parser.add_argument(
        "--input_path",
        required=True,
        type=str,
        help="""Path to a video file if you want to extract frames
            or to a folder of images already extracted by yourself.
            or to a folder of attention images.""",
    )
    parser.add_argument(
        "--output_path",
        default="./",
        type=str,
        help="""Path to store a folder of frames and / or a folder of attention images.
            and / or a final video. Default to current directory.""",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx percent of the mass.""",
    )
    parser.add_argument(
        "--resize",
        default=512,
        type=int,
        nargs="+",
        help="""Apply a resize transformation to input image(s). Use if OOM error.
        Usage (single or W H): --resize 512, --resize 720 1280""",
    )
    parser.add_argument(
        "--video_only",
        action="store_true",
        help="""Use this flag if you only want to generate a video and not all attention images.
            If used, --input_path must be set to the folder of attention images. Ex: ./attention/""",
    )
    parser.add_argument(
        "--fps",
        default=30.0,
        type=float,
        help="FPS of input / output video. Automatically set if you extract frames from a video.",
    )
    parser.add_argument(
        "--video_format",
        default="mp4",
        type=str,
        choices=["mp4", "avi"],
        help="Format of generated video (mp4 or avi).",
    )
    parser.add_argument(
        "--output_filename",
        default="dino-attention",
        type=str,
        help="filename for the attention video.",
    )

    return parser


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    video_generator = VideoGeneratorTF(args)

    video_generator.run()
           