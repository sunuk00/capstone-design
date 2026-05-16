"""
통합 학습 스크립트
마라톤 경로 분할 모델 학습
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # Avoid conflicts with third-party packages named "src" when running as a script.
    sys.modules.pop("src", None)



# src/__init__.py에서 UNet, MarathonSegDataset, collect_pairs, split_pairs, set_seed, run_epoch, EpochStats를 가져옴
from src.core import collect_pairs, split_pairs, set_seed, run_epoch, EpochStats, parse_args_with_config
from src.data import MarathonSegDataset
from src.models import get_model, SegFormerUNet
from src.models.segformer_unet import DeepSupervisionLoss
# src/losses.py에서 BCELoss, BCEIoULoss, BCEDiceLoss를 가져옴
from src.losses import BCELoss, BCEIoULoss, BCEDiceLoss, FocalLoss, FocalDiceLoss, SkeletonRecallLoss, SkeletonRecallDiceLoss, BCEDiceSkelRecallLoss, BoundaryLoss, BCEDiceBoundaryLoss



def build_arg_parser() -> argparse.ArgumentParser:
    """
    커맨드 라인 인자 설정
    """
    parser = argparse.ArgumentParser(description="U-Net training for marathon path segmentation")

    # config 파일 경로. 이 파일의 값이 기본값으로 먼저 적용되고,
    # 동일 옵션을 CLI로 다시 주면 CLI 값이 최종적으로 우선 적용됨.
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    
    # 데이터 관련 인자
    parser.add_argument("--data-root", type=str, default="data/train", help="Training data root directory")
    parser.add_argument("--image-size", type=int, default=512, help="Input image size")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--use-augmentation", action="store_true", help="Apply basic train-time data augmentation")
    parser.add_argument("--use-grayscale-aug", action="store_true", help="Apply grayscale conversion augmentation (50%% prob) to reduce color dependency")
    
    # 모델 관련 인자
    parser.add_argument("--model-name", type=str, default="unet",
                        choices=["unet", "resunet", "deeplabv3", "unet++", "unet3+",
                                 "segformer-b0", "segformer-b2", "segformer-b4",
                                 "segformer_unet-b0", "segformer_unet-b2", "segformer_unet-b4"],
                        help="Model architecture name")
    parser.add_argument("--base-channels", type=int, default=32, help="Base number of channels in U-Net")
    
    # 학습 관련 인자
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (decoder lr for SegFormerUNet)")
    parser.add_argument("--encoder-lr", type=float, default=1e-5,
                        help="Encoder learning rate for SegFormerUNet (사전학습 가중치 보호용). "
                             "SegFormerUNet 이외의 모델에는 적용되지 않는다.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of workers for DataLoader")
    
    # 손실 함수 관련 인자
    parser.add_argument("--loss-type", type=str, default="bce_iou",
                        choices=["bce", "bce_iou", "bce_dice", "focal", "focal_dice", "skel_recall", "skel_recall_dice", "bce_dice_skel", "boundary", "bce_dice_boundary"],
                        help="Loss function type")
    parser.add_argument("--bce-weight", type=float, default=0.5, help="Weight for BCE loss")
    parser.add_argument("--iou-weight", type=float, default=None, help="Weight for IoU loss; defaults to (1 - bce_weight)")
    parser.add_argument("--dice-weight", type=float, default=None, help="Weight for Dice loss; defaults to (1 - bce_weight)")
    parser.add_argument("--pos-weight", type=float, default=10.0, help="Positive weight for class imbalance")
    parser.add_argument("--focal-alpha", type=float, default=0.75, help="Alpha for Focal Loss: weight for positive (path) pixels")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma for Focal Loss: focusing parameter")
    parser.add_argument("--focal-weight", type=float, default=0.5, help="a in: Total Loss = a*Focal + (1-a)*Dice (focal_dice only)")
    parser.add_argument("--skel-weight", type=float, default=0.5, help="a in: Total Loss = a*SkelRecall + (1-a)*Dice (skel_recall_dice only)")
    parser.add_argument("--skel-iters", type=int, default=5, help="Soft skeleton erosion iterations (skel_recall / skel_recall_dice)")
    parser.add_argument("--skel-alpha", type=float, default=0.3, help="BCE weight in bce_dice_skel loss")
    parser.add_argument("--skel-beta", type=float, default=0.3, help="Dice weight in bce_dice_skel loss")
    parser.add_argument("--skel-gamma", type=float, default=0.4, help="SkelRecall weight in bce_dice_skel loss")
    parser.add_argument("--boundary-weight", type=float, default=0.2, help="Boundary loss component weight in bce_dice_boundary")
    parser.add_argument("--boundary-ratio", type=float, default=5.0, help="Upweight multiplier for boundary pixels (boundary & bce_dice_boundary)")
    parser.add_argument("--dilation-radius", type=int, default=3, help="Boundary band thickness in pixels (boundary & bce_dice_boundary)")

    # 출력 관련 인자
    parser.add_argument("--output-dir", type=str, default="outputs/unet_trained", help="Output directory")
    parser.add_argument("--early-stopping-patience", type=int, default=7, help="Stop training after this many epochs without val_loss improvement (0 disables early stopping)")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0, help="Minimum val_loss improvement required to reset early stopping")
    
    return parser



def build_loss_fn(args: argparse.Namespace) -> torch.nn.Module:
    """
    인자에 따라 손실 함수를 생성하는 함수
    """
    if args.loss_type == "bce":
        return BCELoss(pos_weight=args.pos_weight)
    if args.loss_type == "bce_iou":
        return BCEIoULoss(
            bce_weight=args.bce_weight,
            iou_weight=args.iou_weight,
            pos_weight=args.pos_weight,
        )
    elif args.loss_type == "bce_dice":
        return BCEDiceLoss(
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            pos_weight=args.pos_weight,
        )
    elif args.loss_type == "focal":
        return FocalLoss(
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
        )
    elif args.loss_type == "focal_dice":
        return FocalDiceLoss(
            focal_weight=args.focal_weight,
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
        )
    elif args.loss_type == "skel_recall":
        return SkeletonRecallLoss(num_iters=args.skel_iters)
    elif args.loss_type == "skel_recall_dice":
        return SkeletonRecallDiceLoss(
            skel_weight=args.skel_weight,
            num_iters=args.skel_iters,
        )
    elif args.loss_type == "bce_dice_skel":
        return BCEDiceSkelRecallLoss(
            alpha=args.skel_alpha,
            beta=args.skel_beta,
            gamma=args.skel_gamma,
            pos_weight=args.pos_weight,
        )
    elif args.loss_type == "boundary":
        return BoundaryLoss(
            boundary_ratio=args.boundary_ratio,
            dilation_radius=args.dilation_radius,
            pos_weight=args.pos_weight,
        )
    elif args.loss_type == "bce_dice_boundary":
        return BCEDiceBoundaryLoss(
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight if args.dice_weight is not None else 1.0 - args.bce_weight - args.boundary_weight,
            boundary_weight=args.boundary_weight,
            pos_weight=args.pos_weight,
            boundary_ratio=args.boundary_ratio,
            dilation_radius=args.dilation_radius,
        )
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")


def main() -> None:
    # 터미널에서 입력받은 하이퍼파라미터(학습률, 배치사이즈 등)나 파일 경로 등의 설정값을 불러옴 
    args = parse_args_with_config(build_arg_parser)

    # 매번 같은 결과를 얻기 위해 시드 값을 고정함 - 랜덤 시드 고정은 모델의 초기 가중치, 데이터 섞는 순서 등에서 일관된 결과를 얻도록 도와줌
    set_seed(args.seed)

    # 학습에 사용할 images와 경로 이미지 masks, 그리고 학습된 모델이 저장될 출력 폴더의 경로를 설정
    data_root = Path(args.data_root)
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 디렉토리 존재 확인
    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"Expected folders not found: {images_dir} and {masks_dir}")

    print(f"Loading data from {data_root}")

    # 마라톤 이미지와 마스킹 이미지 짝짓기
    pairs = collect_pairs(images_dir, masks_dir)
    if len(pairs) < 2:
        raise RuntimeError("Need at least 2 image-mask pairs to split train/val")

    # Train과 Validation 데이터 분할
    train_pairs, val_pairs = split_pairs(pairs, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Total pairs: {len(pairs)} | Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # 데이터셋 생성
    use_skeleton = args.loss_type == "bce_dice_skel"
    train_ds = MarathonSegDataset(
        train_pairs,
        image_size=args.image_size,
        model_name=args.model_name,
        use_augmentation=args.use_augmentation,
        use_grayscale_aug=args.use_grayscale_aug,
        use_skeleton=use_skeleton,
    )
    val_ds = MarathonSegDataset(
        val_pairs,
        image_size=args.image_size,
        model_name=args.model_name,
        use_skeleton=use_skeleton,
    )

    # DataLoader 생성 - DataLoader는 데이터셋에서 배치 단위로 데이터를 불러오는 역할을 함. 학습 중에 데이터를 섞거나 여러 프로세스를 사용하여 데이터를 불러올 수 있도록 도와줌
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_index)
        print(f"Using device: {device} ({gpu_name})")
    else:
        gpu_name = "cpu"
        print(f"Using device: {device}")

    # 모델 초기화
    model = get_model(
        model_name=args.model_name,
        in_channels=3,
        out_channels=1,
        base_channels=args.base_channels,
    )
    model = model.to(device)

    # 손실 함수 설정
    criterion = build_loss_fn(args) # 인자에 따라 손실 함수를 생성하는 함수
    criterion = criterion.to(device) # 손실 함수를 디바이스로 이동 - 모델과 손실 함수를 같은 디바이스에 올려야 계산이 가능함 : GPU에서 모델을 학습할 때 손실 함수도 GPU로 이동시켜야 함
    # SegFormerUNet deep supervision: 모델이 dict를 반환하므로 래퍼로 감쌈
    if isinstance(model, SegFormerUNet) and model.deep_supervision:
        criterion = DeepSupervisionLoss(criterion)
        print("Loss function: DeepSupervisionLoss (main=1.0, aux3=0.4, aux2=0.2) wrapping", args.loss_type)
    else:
        print(f"Loss function: {args.loss_type}")
    print(f"Model: {args.model_name}")

    # 최적화 함수 설정
    # SegFormerUNet은 사전학습 인코더와 랜덤 초기화 디코더를 차등 학습률로 학습한다.
    # - 인코더(encoder): args.encoder_lr (기본 1e-5) — 사전학습 가중치 보호
    # - 디코더(up1/up2/up3/head 등): args.lr (기본 1e-4) — 랜덤 초기화이므로 높게
    if isinstance(model, SegFormerUNet):
        enc_params = list(model.encoder.parameters())
        enc_param_ids = {id(p) for p in enc_params}
        dec_params = [p for p in model.parameters() if id(p) not in enc_param_ids]
        optimizer = optim.Adam([
            {"params": enc_params, "lr": args.encoder_lr},
            {"params": dec_params, "lr": args.lr},
        ])
        print(f"Optimizer: encoder lr={args.encoder_lr}, decoder lr={args.lr}")
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 학습 기록 저장
    log_history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_path = None
    epochs_without_improvement = 0

    print(f"\nStarting training for {args.epochs} epochs")
    print("-" * 80)

    # Training loop
    for epoch in range(args.epochs):
        # Training
        train_stats = run_epoch(model, train_loader, criterion, device, optimizer)
        
        # Validation
        with torch.no_grad():
            val_stats = run_epoch(model, val_loader, criterion, device, optimizer=None)

        # 통계 기록
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_stats.loss,
            "train_dice": train_stats.dice,
            "train_iou": train_stats.iou,
            "val_loss": val_stats.loss,
            "val_dice": val_stats.dice,
            "val_iou": val_stats.iou,
            "device": str(device),
            "gpu_name": gpu_name,
        }
        log_history.append(log_entry)

        improved = val_stats.loss < (best_val_loss - args.early_stopping_min_delta)
        if improved:
            best_val_loss = val_stats.loss
            best_epoch = epoch + 1
            best_model_path = output_dir / "model_best.pt"
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "val_loss": val_stats.loss,
            }, best_model_path)
        else:
            epochs_without_improvement += 1

        # 출력
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"Train Loss: {train_stats.loss:.6f} | Train Dice: {train_stats.dice:.4f} | Train IoU: {train_stats.iou:.4f} | "
            f"Val Loss: {val_stats.loss:.6f} | Val Dice: {val_stats.dice:.4f} | Val IoU: {val_stats.iou:.4f}"
        )

        if args.early_stopping_patience > 0:
            print(
                f"  Best Val Loss: {best_val_loss:.6f} at epoch {best_epoch} | "
                f"No improvement: {epochs_without_improvement}/{args.early_stopping_patience}"
            )
            if epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"Best val_loss was {best_val_loss:.6f} at epoch {best_epoch}."
                )
                break

    print("-" * 80)
    print("Training completed!")

    # 마지막 epoch 모델 저장
    last_model_path = output_dir / "model_last.pt"
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }, last_model_path)
    print(f"Last model saved to {last_model_path}")

    if best_model_path is not None:
        print(f"Best model: epoch {best_epoch} (val_loss={best_val_loss:.6f}), saved to {best_model_path}")

    # 학습 기록 저장
    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log_history, f, indent=4)
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()
