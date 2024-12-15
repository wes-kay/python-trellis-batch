import os
import uuid
import argparse
import imageio
from PIL import Image, UnidentifiedImageError
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

os.environ['SPCONV_ALGO'] = 'native'

def process_image(file_path, output_dir, resolution=512, texture_size=2048, simplify=0.90):
    try:
        print(f"Processing image: {file_path}")
        image = Image.open(file_path)

        pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        pipeline.cuda()

        outputs = pipeline.run(image, seed=1, formats=["gaussian", "mesh"])
        print(f"Pipeline outputs: {list(outputs.keys())}")

        image_name = os.path.splitext(os.path.basename(file_path))[0]
        sub_folder = os.path.join(output_dir, image_name)
        os.makedirs(sub_folder, exist_ok=True)

        image.save(os.path.join(sub_folder, f"{image_name}.png"))

        if 'gaussian' in outputs and outputs['gaussian']:
            print("Rendering Gaussian video...")
            g = render_utils.render_video(outputs['gaussian'][0], resolution=resolution)
            if 'color' in g:
                imageio.mimsave(os.path.join(sub_folder, f"{image_name}_gs.mp4"), g['color'], fps=30)

        if 'mesh' in outputs and outputs['mesh']:
            print("Rendering Mesh video...")
            m = render_utils.render_video(outputs['mesh'][0], resolution=resolution)
            if 'color' in m:
                imageio.mimsave(os.path.join(sub_folder, f"{image_name}_mesh.mp4"), m['color'], fps=30)
            if 'normal' in m:
                imageio.mimsave(os.path.join(sub_folder, f"{image_name}_mesh_normal.mp4"), m['normal'], fps=30)

        if 'gaussian' in outputs and 'mesh' in outputs:
            print("Generating GLB file...")
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0], 
                outputs['mesh'][0], 
                simplify=simplify, 
                texture_size=texture_size
            )
            glb_path = os.path.join(sub_folder, f"{image_name}.glb")
            glb.export(glb_path)
            os.remove(file_path)
            print(f"GLB exported to: {glb_path}")

    except UnidentifiedImageError as e:
        print(f"Error opening image {file_path}: {e}")
    except KeyError as e:
        print(f"KeyError in outputs: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def process_directory(input_dir, output_dir, resolution, texture_size, simplify):
    while True:
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            continue
        for f in files:
            file_path = os.path.join(input_dir, f)
            if os.path.exists(file_path):
                process_image(file_path, output_dir, resolution, texture_size, simplify)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--texture_size", type=int, default=2048)
    parser.add_argument("--simplify", type=float, default=0.90)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    process_directory(args.input_dir, args.output_dir, args.resolution, args.texture_size, args.simplify)

if __name__ == "__main__":
    main()
