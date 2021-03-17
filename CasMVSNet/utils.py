import os, csv
import numpy as np
import torchvision.utils as vutils
import torch, random
import torch.nn.functional as F


# print arguments
def print_args(args):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


# torch.no_grad warpper for functions
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.to(torch.device("cuda"))
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


def save_scalars(logger, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def save_images(logger, mode, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)


class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input):
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}


# a wrapper to compute metrics for each image individually
def compute_metrics_for_each_image(metric_func):
    def wrapper(depth_est, depth_gt, mask, *args):
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], *args)
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper


@make_nograd_func
@compute_metrics_for_each_image
def Thres_metrics(depth_est, depth_gt, mask, thres):
    assert isinstance(thres, (int, float))
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs(depth_est - depth_gt)
    err_mask = errors > thres
    return torch.mean(err_mask.float())


# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metrics_for_each_image
def AbsDepthError_metrics(depth_est, depth_gt, mask, thres=None):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    error = (depth_est - depth_gt).abs()
    if thres is not None:
        error = error[(error >= float(thres[0])) & (error <= float(thres[1]))]
        if error.shape[0] == 0:
            return torch.tensor(0, device=error.device, dtype=error.dtype)
    return torch.mean(error)

import torch.distributed as dist
def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def reduce_scalar_outputs(scalar_outputs):
    world_size = get_world_size()
    if world_size < 2:
        return scalar_outputs
    with torch.no_grad():
        names = []
        scalars = []
        for k in sorted(scalar_outputs.keys()):
            names.append(k)
            scalars.append(scalar_outputs[k])
        scalars = torch.stack(scalars, dim=0)
        dist.reduce(scalars, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            scalars /= world_size
        reduced_scalars = {k: v for k, v in zip(names, scalars)}

    return reduced_scalars

import torch
from bisect import bisect_right
# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        #print("base_lr {}, warmup_factor {}, self.gamma {}, self.milesotnes {}, self.last_epoch{}".format(
        #    self.base_lrs[0], warmup_factor, self.gamma, self.milestones, self.last_epoch))
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def local_pcd(depth, intr):
    nx = depth.shape[1]  # w
    ny = depth.shape[0]  # h
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    x = x.reshape(nx * ny)
    y = y.reshape(nx * ny)
    p2d = np.array([x, y, np.ones_like(y)])
    p3d = np.matmul(np.linalg.inv(intr), p2d)
    depth = depth.reshape(1, nx * ny)
    p3d *= depth
    p3d = np.transpose(p3d, (1, 0))
    p3d = p3d.reshape(ny, nx, 3).astype(np.float32)
    return p3d

def generate_pointcloud(rgb, depth, ply_file, intr, scale=1.0):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.

    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file

    """
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    points = []
    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            color = rgb[v, u] #rgb.getpixel((u, v))
            Z = depth[v, u] / scale
            if Z == 0: continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))
    file = open(ply_file, "w")
    file.write('''ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            property uchar alpha
            end_header
            %s
            ''' % (len(points), "".join(points)))
    file.close()
    print("save ply, fx:{}, fy:{}, cx:{}, cy:{}".format(fx, fy, cx, cy))

#--------------------------------------------------------------
def read_vissat_idx2name(vissat_path):
    img_idx2name_filename = os.path.join(vissat_path, 'img_idx2name.txt')
    # idx -name dictionary
    idx_name_dict = {}
    with open(img_idx2name_filename, 'r') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=' ')
        for row in read_csv:
            image_idx = int(row[0])
            image_name = row[1]
            idx_name_dict[image_idx] = image_name

    return idx_name_dict


def read_vissat_ref2src(vissat_path):
    ref2src_filename = os.path.join(vissat_path, 'ref2src.txt')

    # ref src_list dictionary
    ref_src_dict = {}
    with open(ref2src_filename, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]
    for i in range(0, len(lines), 3):
        ref_idx = lines[i].split()[-1]
        ref_idx = int(ref_idx)
        src_indices = lines[i + 2].split()[1:]
        src_indices = [int(item) for item in src_indices]
        ref_src_dict[ref_idx] = src_indices

    return ref_src_dict


# def read_vissat_proj_mats(vissat_path):
#     proj_mats_filename = os.path.join(vissat_path, 'proj_mats.txt')
#     proj_mats_dict = {}
#     with open(proj_mats_filename, 'r') as csvfile:
#         read_csv = csv.reader(csvfile, delimiter=' ')
#         for row in read_csv:
#             image_name = row[0]
#             P_4x4 = np.reshape(np.array(row[1:], dtype=np.float32), (4, 4))
#             proj_mats_dict[image_name] = P_4x4
#
#     return proj_mats_dict

def save_vissat_proj_mat(outdir, image_name, cam, append=True):
    proj_mats_filename = os.path.join(outdir, 'proj_mats.txt')
    inv_proj_mats_filename = os.path.join(outdir, 'inv_proj_mats.txt')
    mode = 'a' if append else 'w'
    with open(proj_mats_filename, mode=mode) as fp:
        line = '{} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g}\n'.format(
            image_name, *(cam[0].ravel()))
        fp.write(line)

    with open(inv_proj_mats_filename, mode=mode) as fp:
        line = '{} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g} {:.20g}\n'.format(
            image_name, *(cam[1].ravel()))
        fp.write(line)


def read_vissat_proj_mats(vissat_path, prescaled=True):
    proj_mats_filename = os.path.join(vissat_path, 'proj_mats.txt')
    last_rows_filename = os.path.join(vissat_path, 'last_rows.txt')

    proj_mats_dict = {}
    with open(proj_mats_filename, 'r') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=' ')
        for row in read_csv:
            image_name = row[0]
            P_4x4 = np.reshape(np.array(row[1:], dtype=np.float32), (4, 4))
            proj_mats_dict[image_name] = P_4x4

    if not prescaled:
        # undo prescaling using the last_row info
        last_rows_dict = {}
        with open(last_rows_filename, 'r') as csvfile:
            read_csv = csv.reader(csvfile, delimiter=' ')
            for row in read_csv:
                image_name = row[0]
                last_row = np.array(row[1:], dtype=np.float32)
                last_rows_dict[image_name] = last_row

        for image_name, P_4x4 in proj_mats_dict.items():
            last_row = last_rows_dict[image_name]
            scale = last_row[-1] / P_4x4[-1, -1]
            P_4x4 *= scale
            #P_4x4[:2,:] *= .25
            P_4x4[3,:] = np.array([0,0,0,1])

    return proj_mats_dict

def read_vissat_inv_proj_mats(vissat_path):
    inv_proj_mats_filename = os.path.join(vissat_path, 'inv_proj_mats.txt')
    inv_proj_mats_dict = {}
    with open(inv_proj_mats_filename, 'r') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=' ')
        for row in read_csv:
            image_name = row[0]
            P_4x4 = np.reshape(np.array(row[1:], dtype=np.float32), (4, 4))
            inv_proj_mats_dict[image_name] = P_4x4

    return inv_proj_mats_dict



def read_vissat_reparam_depth(vissat_path):
    reparam_depth_filename = os.path.join(vissat_path, 'reparam_depth.txt')

    depth_min = 1e100
    depth_max = -1e100
    with open(reparam_depth_filename, 'r') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=' ', )
        next(read_csv)
        for row in read_csv:
            print(row)
            d1 = np.float32(row[1])
            d2 = np.float32(row[2])
            for row in read_csv:
                if d1 < depth_min: depth_min = d1
                if d2 > depth_max: depth_max = d2

    return depth_min, depth_max


def read_vissat_aoi_json(vissat_path):
    import json
    with open(os.path.join(vissat_path, '..', '..', 'aoi.json')) as fp:
        aoi = json.load(fp)

    return aoi

def unproject_vissat(depth, proj_4x4, uv=None):
    '''
    :param depth: [rows, cols]
    :param proj_4x4:
    :param uv:  [Nx2] optional, N pixels to unproject, by default all depths are unprojected
                Pixels are input in [col, row]
    :return:
    '''
    # Note: u are cols, v are rows
    width, height = depth.shape[1], depth.shape[0]
    P_inv = proj_4x4[1,:,:]  # o sino np.linalg.inv(proj_4x4[0,:,:])_

    if not uv is None:
        u = uv[:,0]
        v = uv[:,1]
        #d = depth.reshape([-1]) #depth[np.round(v).astype(int).clip(0,depth.shape[0]-1), np.floor(u).astype(int).clip(0,depth.shape[1]-1)]
    else:
        # view u (cols), v (rows)
        u, v = np.meshgrid(np.arange(0, width), np.arange(0, height), indexing='xy')
        u, v = u.reshape([-1]), v.reshape([-1])

    d = depth.ravel()

    uv1m = np.vstack((u,v,np.ones_like(d),d))
    xyzw = P_inv @ uv1m
    xyz = ( xyzw[:3,:] / xyzw[3,:] ).T

    return xyz

def project_vissat(xyz, proj_4x4):
    '''
    :param xyz: shape [Nx3] points in 3D space
    :param proj_4x4:
    :return:
        uv, shape [Nx2] pixels given as [col, row]
    '''
    P = proj_4x4[0,:,:]

    uvst = P @ np.vstack((xyz.T, np.ones((1,xyz.shape[0]))))
    uv = ( uvst[:2,:] / uvst[2,:] ).T
    return uv


def generate_pointcloud_vissat(rgb, depth, ply_file, ref_proj, scale=1.0):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.

    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file

    """
    xyz = unproject_vissat(depth, ref_proj)

    num_points = xyz.shape[0]
    points = np.zeros((num_points, 7))

    u, v = np.meshgrid(np.arange(0, depth.shape[1]), np.arange(0, depth.shape[0]), indexing='xy')
    points[:,:3] = xyz
    points[:,2] /= scale
    points[:,3:6] = rgb[v.ravel(),u.ravel()]

    header = 'ply\n'
    header += 'format ascii 1.0\n'
    header += 'element vertex %d\n' % num_points
    header += 'property float x\n'
    header += 'property float y\n'
    header += 'property float z\n'
    header += 'property uchar red\n'
    header += 'property uchar green\n'
    header += 'property uchar blue\n'
    header += 'property uchar alpha\n'
    header += 'end_header'
    np.savetxt(ply_file, points, fmt=['%f', '%f', '%f', '%d', '%d', '%d', '%d'], delimiter=' ', header=header, comments='')
