import sys

sys.path.append(".")

import torch
import torch.optim as optim
import os, time
from DeepTag.ME_nets.Deeptagnet import Lagrangian_motion_estimate_net
from DeepTag.losses.train_loss import VM_diffeo_loss, NCC
import numpy as np
from DeepTag.data_set.dataset import SimplifiedTrainDataset, create_data_loader
from torch.utils.tensorboard import SummaryWriter
import datetime

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
writer = SummaryWriter(log_dir)

# device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print("device = ", device)


def train_Cardiac_Tagging_ME_net(net,
                                 data_root,
                                 batch_size,
                                 n_epochs,
                                 learning_rate,
                                 model_path,
                                 kl_loss,
                                 recon_loss,
                                 smoothing_loss,
                                 steps_per_epoch=100):
    net.train()
    net.cuda()
    net = net.float()

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # training start time
    training_start_time = time.time()

    # 使用新的数据集创建验证数据加载器
    val_loader = create_data_loader(data_root, 'val', batch_size=batch_size, shuffle=False, num_workers=0)

    train_loss_dict = []
    val_loss_dict = []

    epoch_loss = 0
    train_loss = 0

    for outer_epoch in range(n_epochs):
        print("epochs = ", outer_epoch)
        print("." * 50)

        # 使用新的数据集创建训练数据加载器
        train_loader = create_data_loader(data_root, 'train', batch_size=batch_size, shuffle=True, num_workers=0)
        train_iter = iter(train_loader)

        epoch_loss_0 = 0

        for step in range(steps_per_epoch):
            try:
                seq_paths, cine_imgs = next(train_iter)
            except StopIteration:
                # 如果数据用完了，重新创建迭代器
                train_loader = create_data_loader(data_root, 'train', batch_size=batch_size, shuffle=True,
                                                  num_workers=0)
                train_iter = iter(train_loader)
                seq_paths, cine_imgs = next(train_iter)

            # 移动数据到GPU
            cine_imgs = cine_imgs.to(device).float()

            # 数据预处理
            # cine_imgs 形状: (batch_size, seq_length, height, width)
            shape = cine_imgs.shape
            batch_size_current = shape[0]
            seq_length = shape[1]
            height = shape[2]
            width = shape[3]

            # 确保序列长度至少为2
            if seq_length < 2:
                print(f"序列长度不足，跳过批次: {seq_length}")
                continue

            # 准备训练数据
            # x: 除了第一帧外的其他帧 (moving images)
            # y: 除了最后一帧外的其他帧 (reference images)
            x = cine_imgs[:, 1:, :, :]  # 第2帧到最后一帧
            y = cine_imgs[:, 0:-1, :, :]  # 第1帧到倒数第2帧

            # 调整序列长度
            actual_seq_length = x.shape[1]

            # 重新整形为网络期望的格式
            x = x.contiguous().view(-1, 1, height, width)  # (batch_size * seq_length, 1, height, width)
            y = y.contiguous().view(-1, 1, height, width)  # (batch_size * seq_length, 1, height, width)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            try:
                registered_cine1, registered_cine2, registered_cine1_lag, flow_param, \
                    deformation_matrix, deformation_matrix_neg, deformation_matrix_lag = net(y, x)

                # 计算损失
                train_smoothing_loss = smoothing_loss(deformation_matrix)
                train_smoothing_loss_neg = smoothing_loss(deformation_matrix_neg)
                train_smoothing_loss_lag = smoothing_loss(deformation_matrix_lag)

                # 损失权重
                a = 5  # 平滑损失权重
                b = 1  # 拉格朗日平滑损失权重

                # 总损失
                training_loss = kl_loss(x, flow_param) + \
                                0.5 * recon_loss(x, registered_cine1) + \
                                0.5 * recon_loss(y, registered_cine2) + \
                                0.5 * recon_loss(x, registered_cine1_lag) + \
                                a * train_smoothing_loss + \
                                a * train_smoothing_loss_neg + \
                                b * train_smoothing_loss_lag

                # 检查损失是否过大或为NaN
                if training_loss > 100 or torch.isnan(training_loss):
                    print(f'Loss is too large or NaN, skipping batch: {training_loss}')
                    print(f'Problematic sequence: {seq_paths[0] if seq_paths else "Unknown"}')
                    continue

                # 打印损失详情
                print(f"Step {step}: training_loss: {training_loss:.3f}  "
                      f"recon_loss1: {0.5 * recon_loss(x, registered_cine1):.3f}  "
                      f"recon_loss2: {0.5 * recon_loss(y, registered_cine2):.3f}  "
                      f"recon_loss3: {0.5 * recon_loss(x, registered_cine1_lag):.3f}  "
                      f"kl_loss: {kl_loss(x, flow_param):.3f}  "
                      f"smooth_loss1: {a * train_smoothing_loss:.3f}  "
                      f"smooth_loss2: {a * train_smoothing_loss_neg:.3f}  "
                      f"smooth_loss3: {b * train_smoothing_loss_lag:.3f}")

                # 反向传播
                training_loss.backward()
                optimizer.step()

                # 统计损失
                epoch_loss_0 += training_loss.item()

            except Exception as e:
                print(f"Training step error: {e}")
                print(f"Problematic sequence: {seq_paths[0] if seq_paths else 'Unknown'}")
                continue

        # 计算平均损失
        epoch_loss_0 = epoch_loss_0 / steps_per_epoch
        print(f"Training epoch_loss: {epoch_loss_0:.6f}")

        epoch_loss += epoch_loss_0
        train_loss = epoch_loss / (outer_epoch + 1)
        train_loss_dict.append(train_loss)

        # 保存训练损失
        np.savetxt(os.path.join(model_path, 'train_loss.txt'), train_loss_dict, fmt='%.6f')

        print(f"Average training loss: {train_loss:.6f}")
        writer.add_scalar('training loss', train_loss, outer_epoch)

        # 保存模型检查点
        if outer_epoch % 20 == 0:
            torch.save(net.state_dict(),
                       os.path.join(model_path, f'{outer_epoch:d}_{epoch_loss:.4f}_model.pth'))

        # 验证
        total_val_loss = 0
        val_count = 0

        net.eval()
        with torch.no_grad():
            for val_seq_paths, val_cine_imgs in val_loader:
                try:
                    val_cine_imgs = val_cine_imgs.to(device).float()

                    # 数据预处理（与训练相同）
                    val_shape = val_cine_imgs.shape
                    val_batch_size = val_shape[0]
                    val_seq_length = val_shape[1]
                    val_height = val_shape[2]
                    val_width = val_shape[3]

                    if val_seq_length < 2:
                        continue

                    val_x = val_cine_imgs[:, 1:, :, :]
                    val_y = val_cine_imgs[:, 0:-1, :, :]

                    val_x = val_x.contiguous().view(-1, 1, val_height, val_width)
                    val_y = val_y.contiguous().view(-1, 1, val_height, val_width)

                    # 前向传播
                    val_registered_cine1, val_registered_cine2, val_registered_cine1_lag, \
                        val_flow_param, val_deformation_matrix, val_deformation_matrix_neg, \
                        val_deformation_matrix_lag = net(val_y, val_x)

                    # 计算验证损失
                    val_smoothing_loss = smoothing_loss(val_deformation_matrix)
                    val_smoothing_loss_neg = smoothing_loss(val_deformation_matrix_neg)
                    val_smoothing_loss_lag = smoothing_loss(val_deformation_matrix_lag)

                    a = 5
                    b = 1
                    val_loss = kl_loss(val_x, val_flow_param) + \
                               0.5 * recon_loss(val_x, val_registered_cine1) + \
                               0.5 * recon_loss(val_y, val_registered_cine2) + \
                               0.5 * recon_loss(val_x, val_registered_cine1_lag) + \
                               a * val_smoothing_loss + \
                               a * val_smoothing_loss_neg + \
                               b * val_smoothing_loss_lag

                    val_loss = val_loss / val_batch_size
                    total_val_loss += val_loss.item()
                    val_count += 1

                except Exception as e:
                    print(f"Validation error: {e}")
                    continue

        net.train()  # 切换回训练模式

        # 计算平均验证损失
        if val_count > 0:
            val_epoch_loss = total_val_loss / val_count
            val_loss_dict.append(val_epoch_loss)

            # 保存验证损失
            np.savetxt(os.path.join(model_path, 'val_loss.txt'), val_loss_dict, fmt='%.6f')

            print(f"Validation loss: {val_epoch_loss:.6f}")
            writer.add_scalar('validation loss', val_epoch_loss, outer_epoch)
        else:
            print("No valid validation batches")

    # 保存最终模型
    torch.save(net.state_dict(), os.path.join(model_path, 'end_lagrangian_model.pth'))
    print(f"Training finished! It took {time.time() - training_start_time:.2f}s")


if __name__ == '__main__':
    # 创建模型保存目录
    os.makedirs('./models/Deeptag/', exist_ok=True)
    training_model_path = './models/Deeptag/DeepTag_Lag_tag_256'

    # 数据路径 - 根据您的实际路径修改
    data_path_root = r"/data2/zsn/Dataset/database"

    if not os.path.exists(training_model_path):
        os.makedirs(training_model_path)

    # 训练超参数
    n_epochs = 1500
    learning_rate = 5e-4
    batch_size = 4

    print("......HYPER-PARAMETERS FOR TRAINING......")
    print(f"batch size = {batch_size}")
    print(f"learning rate = {learning_rate}")
    print(f"data path = {data_path_root}")
    print("." * 30)

    # 网络模型
    vol_size = (256, 256)
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 16, 3]
    net = Lagrangian_motion_estimate_net(vol_size, nf_enc, nf_dec)

    # 损失函数
    loss_class = VM_diffeo_loss(image_sigma=0.02, prior_lambda=10, flow_vol_shape=vol_size).cuda()
    my_ncc_loss = NCC()

    # 开始训练
    train_Cardiac_Tagging_ME_net(net=net,
                                 data_root=data_path_root,
                                 batch_size=batch_size,
                                 n_epochs=n_epochs,
                                 learning_rate=learning_rate,
                                 model_path=training_model_path,
                                 kl_loss=loss_class.kl_loss,
                                 recon_loss=my_ncc_loss,
                                 smoothing_loss=loss_class.gradient_loss,
                                 steps_per_epoch=100)