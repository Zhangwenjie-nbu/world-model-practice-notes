# scripts/test_decoder_imagination.py

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# 让脚本可以从项目根目录导入模块
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from envs.grid_nav_env import GridNavEnv
from models.encoder import VisualEncoder
from models.rssm import RSSM
from models.decoder import VisualDecoder
from models.reward_model import RewardModel
from models.common import count_parameters


try:
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_NEAREST = Image.NEAREST


def obs_sequence_to_tensor(obs_sequence: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    将环境图像序列转换为 PyTorch Tensor。

    输入：
        obs_sequence:
            形状为 (T, H, W, C)，dtype=uint8，range=[0,255]

    输出：
        obs_tensor:
            形状为 (1, T, C, H, W)，dtype=float32，range=[0,1]
    """
    assert obs_sequence.ndim == 4
    assert obs_sequence.shape[-1] == 3
    assert obs_sequence.dtype == np.uint8

    tensor = torch.from_numpy(obs_sequence).float() / 255.0

    # T,H,W,C -> T,C,H,W
    tensor = tensor.permute(0, 3, 1, 2)

    # T,C,H,W -> B,T,C,H,W
    tensor = tensor.unsqueeze(0)

    return tensor.to(device)


def encode_sequence(
    encoder: VisualEncoder,
    obs_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    对图像序列逐帧编码。

    输入：
        obs_tensor:
            形状为 (B, T, C, H, W)

    输出：
        embeddings:
            形状为 (B, T, embedding_dim)
    """
    batch_size, seq_len, channels, height, width = obs_tensor.shape

    flat_obs = obs_tensor.reshape(batch_size * seq_len, channels, height, width)
    flat_embeddings = encoder(flat_obs)
    embeddings = flat_embeddings.reshape(batch_size, seq_len, -1)

    return embeddings


def collect_sequence(env: GridNavEnv, seq_len: int = 10):
    """
    从环境中采集一段序列。

    时间步对齐约定：
        obs_0 来自 reset，因此：
            prev_actions[0] = ACTION_STAY
            rewards[0] = 0.0

        obs_t 对应：
            prev_actions[t] = 进入 obs_t 前执行的动作
            rewards[t] = 执行该动作后得到的 reward
    """
    obs_list = []
    prev_action_list = []
    reward_list = []
    done_list = []

    obs = env.reset()

    obs_list.append(obs)
    prev_action_list.append(GridNavEnv.ACTION_STAY)
    reward_list.append(0.0)
    done_list.append(False)

    all_actions = [
        GridNavEnv.ACTION_FORWARD,
        GridNavEnv.ACTION_TURN_LEFT,
        GridNavEnv.ACTION_TURN_RIGHT,
        GridNavEnv.ACTION_STAY,
    ]

    for _ in range(seq_len - 1):
        action = int(np.random.choice(all_actions))

        obs, reward, done, info = env.step(action)

        obs_list.append(obs)
        prev_action_list.append(action)
        reward_list.append(float(reward))
        done_list.append(bool(done))

        if done:
            break

    # 为了测试 shape，提前结束时补齐最后一帧。
    # 正式训练时不这样处理，而是由 replay buffer 按 episode 管理。
    while len(obs_list) < seq_len:
        obs_list.append(obs_list[-1])
        prev_action_list.append(GridNavEnv.ACTION_STAY)
        reward_list.append(0.0)
        done_list.append(True)

    obs_sequence = np.stack(obs_list, axis=0)
    prev_actions = np.array(prev_action_list, dtype=np.int64)
    rewards = np.array(reward_list, dtype=np.float32)
    dones = np.array(done_list, dtype=np.bool_)

    return obs_sequence, prev_actions, rewards, dones


def tensor_images_to_uint8(images: torch.Tensor) -> np.ndarray:
    """
    将模型输出图像转换为 uint8 numpy。

    输入：
        images:
            支持形状：
            - (T, C, H, W)
            - (B, T, C, H, W)

            数值范围应为 [0, 1]。

    输出：
        images_np:
            如果输入是 (T, C, H, W)，输出为 (T, H, W, C)
            如果输入是 (B, T, C, H, W)，输出为 (B, T, H, W, C)
    """
    images = images.detach().cpu().clamp(0.0, 1.0)

    if images.ndim == 4:
        # T,C,H,W -> T,H,W,C
        images = images.permute(0, 2, 3, 1)
    elif images.ndim == 5:
        # B,T,C,H,W -> B,T,H,W,C
        images = images.permute(0, 1, 3, 4, 2)
    else:
        raise ValueError(f"不支持的 image tensor shape: {images.shape}")

    images_np = (images.numpy() * 255.0).round().astype(np.uint8)

    return images_np


def resize_uint8_image(image: np.ndarray, scale: int = 4) -> Image.Image:
    """
    将 uint8 RGB 图像放大，方便在 VS Code 中查看。
    """
    assert image.ndim == 3
    assert image.shape[-1] == 3
    assert image.dtype == np.uint8

    pil_img = Image.fromarray(image, mode="RGB")

    if scale != 1:
        h, w, _ = image.shape
        pil_img = pil_img.resize((w * scale, h * scale), RESAMPLE_NEAREST)

    return pil_img


def save_reconstruction_grid(
    original_images: np.ndarray,
    reconstructed_images: np.ndarray,
    save_path: str,
    scale: int = 4,
):
    """
    保存真实图像与重建图像的对比图。

    第一行：真实图像
    第二行：重建图像
    """
    assert original_images.shape == reconstructed_images.shape
    assert original_images.ndim == 4
    assert original_images.shape[-1] == 3

    seq_len, h, w, c = original_images.shape

    cell_w = w * scale
    cell_h = h * scale
    margin = 4

    canvas_w = seq_len * cell_w + (seq_len - 1) * margin
    canvas_h = 2 * cell_h + margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    for t in range(seq_len):
        x = t * (cell_w + margin)

        original_pil = resize_uint8_image(original_images[t], scale=scale)
        recon_pil = resize_uint8_image(reconstructed_images[t], scale=scale)

        canvas.paste(original_pil, (x, 0))
        canvas.paste(recon_pil, (x, cell_h + margin))

    canvas.save(save_path)


def save_rollout_grid(
    images: np.ndarray,
    save_path: str,
    scale: int = 4,
):
    """
    保存一行 rollout 图像。

    用于可视化 imagined future。
    """
    assert images.ndim == 4
    assert images.shape[-1] == 3
    assert images.dtype == np.uint8

    seq_len, h, w, c = images.shape

    cell_w = w * scale
    cell_h = h * scale
    margin = 4

    canvas_w = seq_len * cell_w + (seq_len - 1) * margin
    canvas_h = cell_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    for t in range(seq_len):
        x = t * (cell_w + margin)
        pil_img = resize_uint8_image(images[t], scale=scale)
        canvas.paste(pil_img, (x, 0))

    canvas.save(save_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"当前设备: {device}")
    print("=" * 60)

    output_dir = ROOT_DIR / "outputs"
    os.makedirs(output_dir, exist_ok=True)

    env = GridNavEnv(
        map_width=10,
        map_height=10,
        image_size=64,
        max_steps=50,
        random_reset=True,
        seed=42,
    )

    seq_len = 10

    obs_sequence, prev_actions, rewards, dones = collect_sequence(
        env=env,
        seq_len=seq_len,
    )

    print("[采集到的环境序列]")
    print(f"obs_sequence shape: {obs_sequence.shape}")
    print(f"prev_actions shape: {prev_actions.shape}")
    print(f"rewards shape: {rewards.shape}")
    print(f"dones shape: {dones.shape}")
    print(f"prev_actions: {prev_actions.tolist()}")
    print(f"rewards: {[round(float(r), 3) for r in rewards.tolist()]}")
    print(f"dones: {dones.tolist()}")

    obs_tensor = obs_sequence_to_tensor(obs_sequence, device)
    actions_tensor = torch.from_numpy(prev_actions).long().unsqueeze(0).to(device)
    reward_targets = torch.from_numpy(rewards).float().view(1, seq_len, 1).to(device)

    print("\n[转换为模型输入]")
    print(f"obs_tensor shape:      {obs_tensor.shape}")
    print(f"actions_tensor shape:  {actions_tensor.shape}")
    print(f"reward_targets shape:  {reward_targets.shape}")

    encoder = VisualEncoder(
        image_size=64,
        in_channels=3,
        embedding_dim=256,
    ).to(device)

    rssm = RSSM(
        embedding_dim=256,
        num_actions=4,
        action_embed_dim=32,
        deter_dim=256,
        stoch_dim=32,
        hidden_dim=256,
        min_std=0.1,
    ).to(device)

    decoder = VisualDecoder(
        feature_dim=288,
        image_size=64,
        out_channels=3,
        hidden_channels=256,
    ).to(device)

    reward_model = RewardModel(
        feature_dim=288,
        hidden_dim=256,
    ).to(device)

    encoder.eval()
    rssm.eval()
    decoder.eval()
    reward_model.eval()

    print("\n[模型参数量]")
    print(f"Encoder 参数量:      {count_parameters(encoder):,}")
    print(f"RSSM 参数量:         {count_parameters(rssm):,}")
    print(f"Decoder 参数量:      {count_parameters(decoder):,}")
    print(f"RewardModel 参数量:  {count_parameters(reward_model):,}")

    with torch.no_grad():
        embeddings = encode_sequence(encoder, obs_tensor)

        posteriors, priors = rssm.observe(
            embeddings=embeddings,
            actions=actions_tensor,
        )

        posterior_feat = rssm.get_feat(posteriors)

        reconstructed = decoder(posterior_feat)

        pred_rewards = reward_model(posterior_feat)

        # 图像重建损失
        # obs_tensor:      (B, T, C, H, W)
        # reconstructed:  (B, T, C, H, W)
        recon_loss = F.mse_loss(reconstructed, obs_tensor)

        # reward 预测损失
        reward_loss = F.mse_loss(pred_rewards, reward_targets)

    print("\n[中间特征]")
    print(f"embeddings shape:       {embeddings.shape}")
    print(f"posterior_feat shape:   {posterior_feat.shape}")

    print("\n[Decoder 重建输出]")
    print(f"reconstructed shape:    {reconstructed.shape}")
    print(f"recon min/max:          {reconstructed.min().item():.4f} / {reconstructed.max().item():.4f}")
    print(f"recon_loss:             {recon_loss.item():.6f}")

    print("\n[RewardModel 输出]")
    print(f"pred_rewards shape:     {pred_rewards.shape}")
    print(f"reward_loss:            {reward_loss.item():.6f}")

    # 保存真实图像 vs 重建图像
    reconstructed_np = tensor_images_to_uint8(reconstructed.squeeze(0))

    recon_save_path = output_dir / "reconstruction_grid.png"
    save_reconstruction_grid(
        original_images=obs_sequence,
        reconstructed_images=reconstructed_np,
        save_path=str(recon_save_path),
        scale=4,
    )

    print("\n[可视化保存]")
    print(f"重建对比图已保存到: {recon_save_path}")
    print("说明：第一行是真实观测，第二行是 Decoder 重建图像。")

    # 从最后一个 posterior state 出发做 imagination rollout
    last_state = rssm.select_state(posteriors, index=-1)

    imagination_horizon = 8

    future_actions = torch.randint(
        low=0,
        high=env.num_actions,
        size=(1, imagination_horizon),
        device=device,
    )

    with torch.no_grad():
        imagined_states = rssm.imagine(
            init_state=last_state,
            actions=future_actions,
        )

        imagined_feat = rssm.get_feat(imagined_states)
        imagined_images = decoder(imagined_feat)
        imagined_rewards = reward_model(imagined_feat)

    imagined_images_np = tensor_images_to_uint8(imagined_images.squeeze(0))

    imagined_save_path = output_dir / "imagined_rollout_grid.png"
    save_rollout_grid(
        images=imagined_images_np,
        save_path=str(imagined_save_path),
        scale=4,
    )

    print("\n[Imagination rollout]")
    print(f"future_actions shape:      {future_actions.shape}")
    print(f"future_actions:            {future_actions.cpu().numpy().tolist()}")
    print(f"imagined_feat shape:       {imagined_feat.shape}")
    print(f"imagined_images shape:     {imagined_images.shape}")
    print(f"imagined_rewards shape:    {imagined_rewards.shape}")
    print(f"imagined reward 示例:      {imagined_rewards.squeeze(0).squeeze(-1).cpu().numpy().round(4).tolist()}")
    print(f"想象 rollout 图已保存到:   {imagined_save_path}")

    assert embeddings.shape == (1, seq_len, 256)
    assert posterior_feat.shape == (1, seq_len, 288)
    assert reconstructed.shape == (1, seq_len, 3, 64, 64)
    assert pred_rewards.shape == (1, seq_len, 1)
    assert imagined_feat.shape == (1, imagination_horizon, 288)
    assert imagined_images.shape == (1, imagination_horizon, 3, 64, 64)
    assert imagined_rewards.shape == (1, imagination_horizon, 1)

    print("\n测试通过：Decoder 重建链路和 imagination rollout 可视化链路均可正常运行。")


if __name__ == "__main__":
    main()