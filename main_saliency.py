import os
import yaml
import torch
from easydict import EasyDict
from src.saliency.mlnet import MLNet, ModMSELoss
from src.DADALoader import DADALoader
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.io import write_video
from src.data_transform import ProcessImages, padding_inv
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import cv2


def parse_configs():
    parser = argparse.ArgumentParser(
        description='PyTorch saliency implementation')
    # For training and testing
    parser.add_argument('--phase', default='train', choices=['train', 'test'],
                        help='Training or testing phase.')
    parser.add_argument('--gpu_id', type=int, default=0, metavar='N',
                        help='The ID number of GPU. Default: 0')
    parser.add_argument('--config', default="cfgs/sac_revere.yml",
                        help='Configuration file for saliceny algorithm.')
    parser.add_argument('--seed', type=int, default=123, metavar='N',
                        help='random seed (default: 123)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.update(vars(args))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cfg.update(device=device)

    return cfg


def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train():
    # prepare output directory
    ckpt_dir = os.path.join(cfg.output, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    # tensorboard logging
    tb_dir = os.path.join(cfg.output, 'tensorboard')
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    TBlogger = SummaryWriter(tb_dir)

    # model
    model = MLNet(cfg.input_shape).to(device)  # ~700MiB

    # dataset loader
    transform_image = transforms.Compose([ProcessImages(cfg.input_shape)])
    transform_salmap = transforms.Compose([ProcessImages(model.output_shape)])
    params_norm = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

    train_data = DADALoader(cfg.data_path, 'training', interval=cfg.frame_interval, max_frames=cfg.max_frames,
                            transforms={'image': transform_image, 'salmap': transform_salmap}, params_norm=params_norm)
    traindata_loader = DataLoader(
        dataset=train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    eval_data = DADALoader(cfg.data_path, 'validation', interval=cfg.frame_interval, max_frames=cfg.max_frames,
                           transforms={'image': transform_image, 'salmap': transform_salmap}, params_norm=params_norm)
    evaldata_loader = DataLoader(
        dataset=eval_data, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    print("# train set: %d, eval set: %d" % (len(train_data), len(eval_data)))

    # loss (criterion)
    criterion = ModMSELoss(model.output_shape).to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,weight_decay=0.0005,momentum=0.9,nesterov=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=1e-4)

    for k in range(cfg.epoch):
        # train the model
        model.train()
        for i, (video_data, salmap_data, _, _) in tqdm(enumerate(traindata_loader), total=len(traindata_loader), desc="Epoch %d [train]" % (k)):
            optimizer.zero_grad()
            # move data to device (gpu)
            video_data = video_data.view(-1, video_data.size(2), video_data.size(3), video_data.size(4)) \
                .contiguous().to(device, dtype=torch.float)  # ~30 MiB
            salmap_data = salmap_data.view(-1, 1, salmap_data.size(3), salmap_data.size(4)) \
                .contiguous().to(device, dtype=torch.float)
            # forward
            out = model.forward(video_data)
            # loss
            loss = criterion(out, salmap_data, model.prior.clone())
            loss.backward()
            optimizer.step()
            # print
            TBlogger.add_scalars(
                "loss", {'train_loss': loss.item()}, k * len(traindata_loader) + i)
            # print("batch: %d / %d, train loss = %.3f"%(i, len(traindata_loader), loss.item()))

        # eval the model
        model.eval()
        loss_val = 0
        for i, (video_data, salmap_data, _, _) in tqdm(enumerate(evaldata_loader), total=len(evaldata_loader), desc="Epoch %d [eval]" % (k)):
            # move data to device (gpu)
            video_data = video_data.view(-1, video_data.size(2), video_data.size(3), video_data.size(4)) \
                .contiguous().to(device, dtype=torch.float)  # ~30 MiB
            salmap_data = salmap_data.view(-1, 1, salmap_data.size(3), salmap_data.size(4)) \
                .contiguous().to(device, dtype=torch.float)
            with torch.no_grad():
                # forward
                out = model.forward(video_data)
                loss = criterion(out, salmap_data, model.prior.clone())
                loss_val += loss.item()
        loss_val /= i
        # write tensorboard logging
        TBlogger.add_scalars(
            "loss", {'eval_loss': loss_val}, (k+1) * len(traindata_loader))

        # save the model
        model_file = os.path.join(ckpt_dir, 'saliency_model_%02d.pth' % (k))
        torch.save({'epoch': k,
                    'model': model.module.state_dict() if len(gpu_ids) > 1 else model.state_dict(),
                    'optimizer': optimizer.state_dict()}, model_file)
    TBlogger.close()


def test():
    # prepare result path
    result_dir = os.path.join(cfg.output, 'testing')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # testing dataset
    transform_image = transforms.Compose([ProcessImages(cfg.input_shape)])
    params_norm = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    test_data = DADALoader(cfg.data_path, 'testing', transforms={ 'image': transform_image, 'salmap': None}, use_salmap=False, use_fixation=False, params_norm=params_norm)
    testdata_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    # model
    model = MLNet(cfg.input_shape).to(device)  # ~700MiB

    # load model weight file
    model_file = cfg.sal_model
    ckpt = torch.load(model_file, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    # run inference
    with torch.no_grad():
        for i, (video_data, salmap_data, coord_data, data_info) in enumerate(testdata_loader):
            # parse data info
            data_info = data_info.cpu().numpy() if data_info.is_cuda else data_info.detach().numpy()
            filename = str(int(data_info[0, 0])) + \
                '_%03d' % (int(data_info[0, 1])) + '.avi'
            num_frames, height, width = data_info[0, 2:].astype(np.int)

            # prepare result video writer
            result_videofile = os.path.join(result_dir, filename)
            if os.path.exists(result_videofile):
                continue
            # for each video
            pred_video = []
            for fid in tqdm(range(num_frames), total=num_frames, desc="Testing video [%d / %d]" % (i, len(testdata_loader))):
                frame_data = video_data[:, fid].to(
                    device, dtype=torch.float)  # (B, C, H, W)
                # forward
                out = model.forward(frame_data)
                out = out.cpu().numpy() if out.is_cuda else out.detach().numpy()
                out = np.squeeze(out)  # (60, 80)
                # decode results
                pred_saliency = padding_inv(out, height, width)
                pred_saliency = np.tile(np.expand_dims(
                    np.uint8(pred_saliency), axis=-1), (1, 1, 3))
                pred_video.append(pred_saliency)

            pred_video = np.array(pred_video, dtype=np.uint8)  # (T, H, W, C)
            write_video(result_videofile, torch.from_numpy(
                pred_video), test_data.fps)


def eval(pred_dir):
    # import metrics.saliency.saliency_metrics as metrics
    import transplant
    matlab = transplant.Matlab(jvm=False, desktop=False)
    matlab.addpath('metrics/saliency/code_forMetrics')

    metrics_all = []
    for filename in sorted(os.listdir(pred_dir)):
        if not filename.endswith('.avi'):
            continue
        # read predicted saliency video
        salmaps_pred = read_saliency_videos(os.path.join(pred_dir, filename))
        # read ground truth video
        salmaps_gt = read_saliency_videos(os.path.join(
            cfg.data_path, 'testing', 'salmap_videos', filename.split('_')[0], filename.split('_')[1]))
        assert salmaps_pred.shape[0] == salmaps_gt.shape[0], "Predictions and GT are not aligned! %s" % (
            filename)
        # compute metrics for each frame
        num_frames = salmaps_pred.shape[0]
        metrics_video = np.zeros((num_frames, 4), dtype=np.float32)
        for i, (map_pred, map_gt) in tqdm(enumerate(zip(salmaps_pred, salmaps_gt)), total=salmaps_pred.shape[0], desc="Evaluate %s" % (filename)):
            # We cannot compute AUC metrics (AUC-Judd, shuffled AUC, and AUC_borji)
            # since we do not have binary map of human fixation points
            sim = matlab.similarity(map_pred, map_gt)
            cc = matlab.CC(map_pred, map_gt)
            nss = matlab.NSS(map_pred, map_gt)
            kl = matlab.KLdiv(map_pred, map_gt)
            metrics_video[i, :] = np.array(
                [sim, cc, nss, kl], dtype=np.float32)
    metrics_all.append(metrics_video)
    return metrics_all


def evaluate():
    pred_dir = os.path.join(cfg.output, 'testing')
    assert os.path.exists(pred_dir), "No predicted results!"

    result_file = os.path.join(cfg.output, 'eval_mlnet.npy')
    if not os.path.exists(result_file):
        # run evaluation
        metrics_all = eval(pred_dir)
        np.save(result_file, metrics_all)
    else:
        metrics_all = np.load(result_file)
    metrics_all = np.array(metrics_all, dtype=np.float32)
    eval_result = np.mean(metrics_all, axis=(0, 1))

    # report performances
    from terminaltables import AsciiTable
    display_data = [["Metrics", "SIM", "CC", "NSS", "KL"], ["Ours"]]
    for val in eval_result:
        display_data[1].append("%.3f" % (val))
    display_title = "Video Saliency Prediction Results on DADA-2000 Dataset."
    table = AsciiTable(display_data, display_title)
    table.inner_footing_row_border = True
    print(table.table)


def read_saliency_videos(video_file):
    assert os.path.exists(
        video_file), "Saliency video file does not exist! %s" % (video_file)
    salmaps = []
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    while (ret):
        # RGB (660, 1584, 3) --> Gray (660, 1584)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        salmaps.append(frame)
        ret, frame = cap.read()
    salmaps = np.array(salmaps, dtype=np.float32) / 255.0
    return salmaps


if __name__ == "__main__":

    # parse input arguments
    cfg = parse_configs()

    # fix random seed
    set_deterministic(cfg.seed)

    # gpu options
    print("Using GPU devices: ", cfg.gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    if cfg.phase == 'train':
        train()
    elif cfg.phase == 'test':
        test()
    elif cfg.phase == 'eval':
        evaluate()
    else:
        raise NotImplementedError
