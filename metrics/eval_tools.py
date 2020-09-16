import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline


def draw_pr_curves(precisions, recalls, time_to_accidents, legend_text, vis_file):
    plt.figure(figsize=(10,5))
    fontsize = 18
    plt.plot(recalls, precisions, 'r-')
    plt.axvline(x=0.8, ymax=1.0, linewidth=2.0, color='k', linestyle='--')
    plt.xlabel('Recall', fontsize=fontsize)
    plt.ylabel('Precision', fontsize=fontsize)
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.title('Precision Recall Curves', fontsize=fontsize)
    plt.legend([legend_text], fontsize=fontsize)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(vis_file)


# def compute_metrics(all_pred, all_toas, all_fps, thresh):
#     Ntp, Ntn, Nfp, Nfn = 0, 0, 0, 0
#     tta = 0  # time to accident
#     Nvid_tp = 0  # number of true positive videos
#     # iterate results of each video
#     for vid, pred in enumerate(all_pred):
#         n_frames = pred.shape[0]
#         toa = all_toas[vid]
#         fps = all_fps[vid]
#         pos = np.sum((pred > thresh).astype(np.int))
#         if toa < n_frames and toa >= 0:
#             Ntp += pos             # true positive
#             Nfn += n_frames - pos  # false negative
#             if pos > 0:
#                 # time of accident (clipped to larger than 0), unit: second
#                 tta += np.maximum(0, toa - np.where(pred > thresh)[0][0]) / fps
#                 Nvid_tp += 1
#         else:
#             Nfp += pos             # false positive
#             Ntn += n_frames - pos  # true negative
    
#     precision = Ntp / (Ntp + Nfp) if Ntp + Nfp > 0 else 0
#     recall = Ntp / (Ntp + Nfn) if Ntp + Nfn > 0 else 0
#     mtta = tta / Nvid_tp if Nvid_tp > 0 else 0
#     return precision, recall, mtta


# def evaluation_accident(all_pred, all_labels, all_toas, all_fps, draw_curves=False, vis_file=None):
#     """
#     all_pred: list, (N_videos, N_frames) a list of ndarray
#     all_labels: list, (N_videos,)
#     all_toas: list, (N_videos)
#     all_fps: list, (N_videos)
#     """
#     # find the minimum predicted score
#     min_pred = np.inf
#     for pred in all_pred:
#         if min_pred > np.min(pred):
#             min_pred = np.min(pred)

#     # thresholds = np.arange(0, 1.01, 0.01)
#     thresholds = np.arange(max(min_pred, 0), 1.0, 0.001)
#     precisions = np.zeros((len(thresholds)), dtype=np.float32)
#     recalls = np.zeros((len(thresholds)), dtype=np.float32)
#     mean_toas = np.zeros((len(thresholds)), dtype=np.float32)

#     for i, thresh in enumerate(thresholds):
#         precisions[i], recalls[i], mean_toas[i] = compute_metrics(all_pred, all_toas, all_fps, thresh)

#     # sort the results with recall
#     inds = np.argsort(recalls)
#     precisions = precisions[inds]
#     recalls = recalls[inds]
#     mean_toas = mean_toas[inds]

#     # unique the recall
#     new_recalls, indices = np.unique(recalls, return_index=True)
#     # for each unique recall, get the best tta and precision
#     new_precisions = np.zeros_like(new_recalls)
#     new_toas = np.zeros_like(new_recalls)
#     for i in range(len(indices)-1):  # first N-1 values
#         new_precisions[i] = np.max(precisions[indices[i]:indices[i+1]])
#         new_toas[i] = np.max(mean_toas[indices[i]:indices[i+1]])
#     new_precisions[-1] = precisions[indices[-1]]
#     new_toas[-1] = mean_toas[indices[-1]]

#     # compute average precision (AP) score
#     AP = 0.0
#     if new_recalls[0] != 0:
#         AP += new_precisions[0]*(new_recalls[0]-0)
#     for i in range(1,len(new_precisions)):
#         # compute the area under the P-R curve
#         AP += (new_precisions[i-1] + new_precisions[i]) * (new_recalls[i] - new_recalls[i-1]) / 2
#     # mean Time to Accident (mTTA)
#     mTTA = np.mean(new_toas)
#     # TTA at 80% recall
#     sort_time = new_toas[np.argsort(new_recalls)]
#     sort_recall = np.sort(new_recalls)
#     TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))]

#     if draw_curves:
#         assert vis_file is not None, "vis_file is not specified!"
#         legend_text = 'SAC Gaussian (AP=%.2f%%)'%(AP * 100)
#         draw_pr_curves(new_precisions, new_recalls, new_toas, legend_text, vis_file)
    
#     p05, r05, t05 = compute_metrics(all_pred, all_toas, all_fps, 0.5)
#     print("\nprecision@0.5 = %.4f, recall@0.5 = %.4f, TTA@0.5 = %.4f\n"%(p05, r05, t05))
#     return AP, mTTA, TTA_R80

def compute_metrics(preds_eval, all_labels, time_of_accidents, thresolds):

    Precision = np.zeros((len(thresolds)))
    Recall = np.zeros((len(thresolds)))
    Time = np.zeros((len(thresolds)))
    cnt = 0
    for Th in thresolds:
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0  # number of TP videos
        # iterate each video sample
        for i in range(len(preds_eval)):
            # true positive frames: (pred->1) * (gt->1)
            tp =  np.where(preds_eval[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                # if at least one TP, compute the relative (1 - rTTA)
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter+1
            # all positive frames
            Tp_Fp += float(len(np.where(preds_eval[i]>=Th)[0])>0)
        if Tp_Fp == 0:  # predictions of all videos are negative
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0: # gt of all videos are negative
            continue
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1
    return Precision, Recall, Time

def evaluation_accident(all_pred, all_labels, time_of_accidents, fps=30.0):
    """
    :param: all_pred (N x T), where N is number of videos, T is the number of frames for each video
    :param: all_labels (N,)
    :param: time_of_accidents (N,) int element
    :output: AP (average precision, AUC), mTTA (mean Time-to-Accident), TTA@R80 (TTA at Recall=80%)
    """

    preds_eval = []
    min_pred = np.inf
    # n_frames = 0
    for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx] > 0:
            pred = all_pred[idx, :int(toa)]  # positive video
        else:
            pred = all_pred[idx, :]  # negative video
        # find the minimum prediction
        min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
        preds_eval.append(pred)
        # n_frames += len(pred)
    total_seconds = all_pred.shape[1] / fps

    # compute precision, recall, and tta for each threshold
    thresolds = np.arange(max(min_pred, 0), 1.0, 0.001)
    Precision, Recall, Time = compute_metrics(preds_eval, all_labels, time_of_accidents, thresolds)
    # when threshold=0.5
    p05, r05, t05 = compute_metrics(preds_eval, all_labels, time_of_accidents, [0.5])

    # sort the metrics with recall (ascending)
    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    # unique the recall, and fetch corresponding precisions and TTAs
    _,rep_index = np.unique(Recall,return_index=1)
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
    # sort by descending order
    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    # compute AP (area under P-R curve)
    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    # transform the relative mTTA to seconds
    mTTA = np.mean(new_Time) * total_seconds
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))] * total_seconds

    return AP, mTTA, TTA_R80, p05[0], r05[0], t05[0] * total_seconds


def evaluation_fixation(preds, labels, metric='mse'):
    """Evaluate the Mean Squared Error for fixation prediction
    """
    mse_result = []
    for i, gt_fixes in enumerate(labels):
        inds = np.where(gt_fixes[:, 0] > 0)[0]
        if len(inds) > 0:  # ignore the non-accident frames
            pred_fix = preds[i][inds, :]
            gt_fix = gt_fixes[inds, :]
            mse = np.mean(np.sqrt(np.sum(np.square(pred_fix - gt_fix), axis=1)), axis=0)
            mse_result.append(mse)
    mse_result = np.array(mse_result, dtype=np.float32)
    mse_final = np.mean(mse_result)
    return mse_final


def print_results(Epochs, APvid_all, AP_all, mTTA_all, TTA_R80_all, Unc_all, result_dir):
    result_file = os.path.join(result_dir, 'eval_all.txt')
    with open(result_file, 'w') as f:
        for e, APvid, AP, mTTA, TTA_R80, Un in zip(Epochs, APvid_all, AP_all, mTTA_all, TTA_R80_all, Unc_all):
            f.writelines('Epoch: %s,'%(e) + ' APvid={:.3f}, AP={:.3f}, mTTA={:.3f}, TTA_R80={:.3f}, mAU={:.5f}, mEU={:.5f}\n'.format(APvid, AP, mTTA, TTA_R80, Un[0], Un[1]))
    f.close()


def vis_results(vis_data, batch_size, vis_dir, smooth=False, vis_batchnum=2):
    assert vis_batchnum <= len(vis_data)
    for b in range(vis_batchnum):
        results = vis_data[b]
        pred_frames = results['pred_frames']
        labels = results['label']
        toa = results['toa']
        video_ids = results['video_ids']
        detections = results['detections']
        uncertainties = results['pred_uncertain']
        for n in range(batch_size):
            pred_mean = pred_frames[n, :]  # (90,)
            pred_std_alea = 1.0 * np.sqrt(uncertainties[n, :, 0])
            pred_std_epis = 1.0 * np.sqrt(uncertainties[n, :, 1])
            xvals = range(len(pred_mean))
            if smooth:
                # sampling
                xvals = np.linspace(0,len(pred_mean)-1,20)
                pred_mean_reduce = pred_mean[xvals.astype(np.int)]
                pred_std_alea_reduce = pred_std_alea[xvals.astype(np.int)]
                pred_std_epis_reduce = pred_std_epis[xvals.astype(np.int)]
                # smoothing
                xvals_new = np.linspace(1,len(pred_mean)+1,80)
                pred_mean = make_interp_spline(xvals, pred_mean_reduce)(xvals_new)
                pred_std_alea = make_interp_spline(xvals, pred_std_alea_reduce)(xvals_new)
                pred_std_epis = make_interp_spline(xvals, pred_std_epis_reduce)(xvals_new)
                pred_mean[pred_mean >= 1.0] = 1.0-1e-3
                xvals = xvals_new
                # fix invalid values
                indices = np.where(xvals <= toa[n])[0]
                xvals = xvals[indices]
                pred_mean = pred_mean[indices]
                pred_std_alea = pred_std_alea[indices]
                pred_std_epis = pred_std_epis[indices]
            # plot the probability predictions
            fig, ax = plt.subplots(1, figsize=(24, 3.5))
            ax.fill_between(xvals, pred_mean - pred_std_alea, pred_mean + pred_std_alea, facecolor='wheat', alpha=0.5)
            ax.fill_between(xvals, pred_mean - pred_std_epis, pred_mean + pred_std_epis, facecolor='yellow', alpha=0.5)
            plt.plot(xvals, pred_mean, linewidth=3.0)
            if toa[n] <= pred_frames.shape[1]:
                plt.axvline(x=toa[n], ymax=1.0, linewidth=3.0, color='r', linestyle='--')
            # plt.axhline(y=0.7, xmin=0, xmax=0.9, linewidth=3.0, color='g', linestyle='--')
            # draw accident region
            x = [toa[n], pred_frames.shape[1]]
            y1 = [0, 0]
            y2 = [1, 1]
            ax.fill_between(x, y1, y2, color='C1', alpha=0.3, interpolate=True)
            fontsize = 25
            plt.ylim(0, 1.1)
            plt.xlim(1, pred_frames.shape[1])
            plt.ylabel('Probability', fontsize=fontsize)
            plt.xlabel('Frame (FPS=20)', fontsize=fontsize)
            plt.xticks(range(0, pred_frames.shape[1], 10), fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.grid(True)
            plt.tight_layout()
            tag = 'pos' if labels[n] > 0 else 'neg'
            plt.savefig(os.path.join(vis_dir, video_ids[n] + '_' + tag + '.png'))
            plt.close()
            # plt.show()