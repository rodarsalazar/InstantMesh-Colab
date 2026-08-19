"""Example: process a folder of images in batch using InstantMeshProcessor.

Usage (in Colab or locally after installing dependencies):

from examples.batch_process_example import run_example
run_example('/path/to/images', '/content/outputs', config='configs/instant-mesh-large.yaml')

This script demonstrates how to process ~50 images efficiently by
initializing the models once and reusing them across the loop.
"""

import os
from glob import glob
from run import InstantMeshProcessor


def run_example(input_folder: str, output_folder: str, config: str = 'configs/instant-mesh-large.yaml'):
    # gather up to 50 images
    imgs = sorted([p for p in glob(os.path.join(input_folder, '*')) if p.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])[:50]
    os.makedirs(output_folder, exist_ok=True)

    proc = InstantMeshProcessor(config)
    proc.load_models()

    results = proc.process_images(imgs, output_folder, diffusion_steps=50, remove_bg=True, export_texmap=False, save_video_flag=False)

    print('Processed', len(results), 'images')
    for r in results:
        print(r)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default='configs/instant-mesh-large.yaml')
    args = parser.parse_args()
    run_example(args.input, args.output, args.config)
