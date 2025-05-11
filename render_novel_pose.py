import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
import numpy as np
from scripts.posmap_generator.lib.renderer import gl
from utils.general_utils import safe_state, to_cuda
from argparse import ArgumentParser
from arguments import ModelParams, get_combined_args, NetworkParams, OptimizationParams
from model.avatar_model import AvatarModel
from scene.dataset_mono import MonoDataset_novel_pose
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


def change_pose(novel_pose_dataset: MonoDataset_novel_pose, pose_file: str):
    new_smpl = np.load(pose_file)
    new_pose = torch.from_numpy(new_smpl['body_pose'])
    global_orient = torch.from_numpy(new_smpl['global_orient'])
    new_transl = torch.from_numpy(new_smpl['transl'])
    rotvec_y = torch.tensor([0, 0, torch.pi])
    rotvec_x = torch.tensor([torch.pi, 0, 0])
    rotvec_z = torch.tensor([0, 0, torch.pi])

    # Convert to rotation matrices
    R_z = axis_angle_to_matrix(rotvec_z)
    R_y = axis_angle_to_matrix(rotvec_y)
    R_x = axis_angle_to_matrix(rotvec_x)

    # Combine correction
    R_correction = R_z @ R_y @ R_x  
    # Convert global_orient axis-angle to rotation matrices
    R_global = axis_angle_to_matrix(global_orient)  # [num_frames, 3, 3]

    # Apply correction: new_R = R_correction @ R_global
    R_fixed = R_correction @ R_global  # [num_frames, 3, 3]

    # Convert back to axis-angle
    global_orient = matrix_to_axis_angle(R_fixed)  # [num_frames, 3]
    new_pose = torch.cat((global_orient, new_pose), dim=1)  # [num_frames, 72]
    # put the global orientation in the first 3 elements
    novel_pose_dataset.pose_data = new_pose
    novel_pose_dataset.transl_data = new_transl
    return novel_pose_dataset


def render_sets(model: ModelParams, net, opt, epoch: int, pose_file: str):
    with torch.no_grad():
        avatarmodel = AvatarModel(model, net, opt, train=False)
        avatarmodel.training_setup()
        avatarmodel.load(epoch)

        novel_pose_dataset = avatarmodel.getNovelposeDataset()
        novel_pose_dataset = change_pose(
            novel_pose_dataset, pose_file)
        novel_pose_name = os.path.basename(os.path.dirname(pose_file))
        novel_pose_loader = torch.utils.data.DataLoader(novel_pose_dataset,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        num_workers=4,)

        render_path = os.path.join(
            avatarmodel.model_path, 'novel_pose', f"ours_{epoch}_empose_{novel_pose_name}")
        makedirs(render_path, exist_ok=True)
        assert model.smpl_type == 'smpl'
        for idx, batch_data in enumerate(tqdm(novel_pose_loader, desc="Rendering progress")):
            batch_data = to_cuda(batch_data, device=torch.device('cuda:0'))
            image, = avatarmodel.render_free_stage1(batch_data, 59400)

            torchvision.utils.save_image(image, os.path.join(
                render_path, '{0:05d}'.format(idx) + ".png"))


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    network = NetworkParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--epoch", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--pose_file", required=True,
                    help="Path to the .npz file containing the novel pose")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    safe_state(args.quiet)

    render_sets(model.extract(args), network.extract(
        args), op.extract(args), args.epoch, args.pose_file)
