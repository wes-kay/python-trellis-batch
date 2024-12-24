import os
import argparse
import imageio
from PIL import Image, UnidentifiedImageError
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import time
os.environ['SPCONV_ALGO'] = 'native'

def process_image(file_path, output_dir):
    try:
        print(f"Processing image: {file_path}")
        image = Image.open(file_path)
        
        image_name = os.path.splitext(os.path.basename(file_path))[0]
        sub_folder = os.path.join(output_dir, image_name)
        os.makedirs(sub_folder, exist_ok=True)

        pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        pipeline.cuda()

        outputs = pipeline.run(image, seed=1, formats=["gaussian", "mesh"])
        print(f"Pipeline outputs: {list(outputs.keys())}")

        image.save(os.path.join(sub_folder, os.path.basename(file_path)))

        if 'gaussian' in outputs and outputs['gaussian']:
            print("Rendering Gaussian video...")
            g = render_utils.render_video(outputs['gaussian'][0], resolution=512)
            if 'color' in g:
                p = os.path.join(sub_folder, f"{image_name}_gaussian.mp4")
                imageio.mimsave(p, g['color'], fps=30)

        if 'mesh' in outputs and outputs['mesh']:
            print("Rendering Mesh video...")
            m = render_utils.render_video(outputs['mesh'][0], resolution=512)
            if 'color' in m:
                p = os.path.join(sub_folder, f"{image_name}_mesh.mp4")
                imageio.mimsave(p, m['color'], fps=30)
            if 'normal' in m:
                p = os.path.join(sub_folder, f"{image_name}_mesh_normal.mp4")
                imageio.mimsave(p, m['normal'], fps=30)

        if 'gaussian' in outputs and 'mesh' in outputs:
            print("Generating GLB file...")
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0], 
                outputs['mesh'][0], 
                simplify=0.90, 
                texture_size=2048
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

def process_directory(input_dir, output_dir):
    while True:
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            time.sleep(1000)
            continue
        for f in files:
            file_path = os.path.join(input_dir, f)
            if os.path.exists(file_path):
                process_image(file_path, output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
