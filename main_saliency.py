import os
import torch
from src.saliency.mlnet import MLNet
from src.TEDLoader import TEDLoader
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.io import write_video
from src.data_transform import ProcessImages, padding_inv
import numpy as np
from tqdm import tqdm


MODEL_PATH = "models/saliency/mlnet_25.pth"
INPUT_SHAPE = [480, 640]
FPS = 10


def parse_arguments():
    parser = argparse.ArgumentParser(description='Saliency implementation')
    parser.add_argument('data_path', help='')
    parser.add_argument('output_path', help='')
    parser.add_argument('--gpu_id', type=int, default=0, metavar='N', help='')
    return parser.parse_args()


def main(data_path, output_path, device):
    os.makedirs(output_path, exist_ok=True)

    transform_image = transforms.Compose([ProcessImages(INPUT_SHAPE)])
    params_norm = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    test_data = TEDLoader(data_path, transforms=transform_image, params_norm=params_norm)
    testdata_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    model = MLNet(INPUT_SHAPE).to(device)  # ~700MiB

    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (video, video_info) in enumerate(tqdm(testdata_loader, total=len(testdata_loader), desc="Creating heatmap videos")):
            video_id = str(video_info[0][0])
            num_frames, height, width = video_info[1:]
            num_frames = num_frames.item()
            height = height.item()
            width = width.item()

            result_dir = os.path.join(output_path, video_id)
            os.makedirs(result_dir, exist_ok=True)

            result_videofile = os.path.join(result_dir, f"{video_id}_heatmap.avi")

            if os.path.exists(result_videofile):
                continue

            pred_video = []
            for fid in range(num_frames):
                frame_data = video[:, fid].to(device, dtype=torch.float)

                # Forward
                out = model(frame_data)
                out = out.cpu().numpy() if out.is_cuda else out.detach().numpy()
                out = np.squeeze(out)

                # Decode results and convert to RGB for visualization
                pred_saliency = padding_inv(out, height, width)
                pred_saliency = np.tile(np.expand_dims(np.uint8(pred_saliency * 255), axis=-1), (1, 1, 3))  # Convert to RGB
                pred_video.append(pred_saliency)

            pred_video = np.array(pred_video, dtype=np.uint8)  # (T, H, W, C)
            write_video(result_videofile, torch.from_numpy(pred_video), FPS)


if __name__ == "__main__":
    args = parse_arguments()

    print("Using GPU devices: ", args.gpu_id)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    main(args.data_path, args.output_path, device)
