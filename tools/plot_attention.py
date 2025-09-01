#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors


def load_attention(attn_dir: str, layer: int, head: int) -> np.ndarray:
    path = os.path.join(attn_dir, f"layer{layer}_head{head}_attn.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    attn = np.load(path)
    return attn.astype(np.float32)


def load_neff(attn_dir: str, layer: int) -> np.ndarray:
    path = os.path.join(attn_dir, f"layer{layer}_per_position_neff.npy")
    if not os.path.exists(path):
        return None
    neff = np.load(path)
    return neff.astype(np.float32)


def select_region(attn: np.ndarray, region: str, size: int) -> np.ndarray:
    n = attn.shape[0]
    if region == "full" or size <= 0 or size >= n:
        return attn
    if region == "head":  # 前 size tokens
        return attn[:size, :size]
    if region == "tail":  # 后 size tokens
        return attn[-size:, -size:]
    if region == "center":
        half = size // 2
        mid = n // 2
        s, e = max(0, mid - half), min(n, mid + half)
        return attn[s:e, s:e]
    if region == "band":  # 对角带状显示（近邻注意）
        # 用一个掩码只保留对角附近 +-k 的元素，其它置0
        k = size
        out = np.zeros_like(attn)
        for i in range(n):
            j0, j1 = max(0, i - k), min(n, i + k + 1)
            out[i, j0:j1] = attn[i, j0:j1]
        return out
    return attn


essential_cmaps = {
    "attn": "viridis",
    "neff": "magma",
}


def plot_attention(attn: np.ndarray, out_path: str, title: str = None, cmap: str = None, dpi: int = 200, annot: bool = False, log=False, clip_pct=0.0):
    plt.figure(figsize=(8, 7), dpi=dpi)
    data = attn.copy()
    if clip_pct and clip_pct > 0:
        lo = np.quantile(data[data>0], clip_pct/100.0) if (data>0).any() else 0
        hi = np.quantile(data, 1-clip_pct/100.0)
        data = np.clip(data, lo, hi)
    if log:
        eps = 1e-8
        data = np.log10(np.maximum(data, eps))
        vmin, vmax = data.min(), data.max()
        im = plt.imshow(data, aspect='auto', interpolation='nearest', cmap=cmap or essential_cmaps["attn"], vmin=vmin, vmax=vmax)
    else:
        im = plt.imshow(data, aspect='auto', interpolation='nearest', cmap=cmap or essential_cmaps["attn"], vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    if title:
        plt.title(title)
    plt.xlabel("Key position")
    plt.ylabel("Query position")

    # 当区域较小且开启annot时，标注数值
    h, w = attn.shape
    if annot and h <= 64 and w <= 64:
        # 选择阈值决定字体颜色
        for i in range(h):
            for j in range(w):
                val = attn[i, j]
                color = 'white' if val > 0.5 else 'black'
                plt.text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_neff_curve(neff_vec: np.ndarray, out_path: str, title: str = None, dpi: int = 160):
    plt.figure(figsize=(9, 3), dpi=dpi)
    plt.plot(neff_vec, lw=1.2)
    plt.grid(alpha=0.3)
    if title:
        plt.title(title)
    plt.xlabel("Position (query)")
    plt.ylabel("Neff")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot attention heatmap and per-position Neff")
    parser.add_argument('--attn_dir', type=str, default='attn_dump', help='Directory containing *.npy dumps')
    parser.add_argument('--out_dir', type=str, default='attn_figs', help='Output directory for figures')
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--head', type=int, default=0)
    parser.add_argument('--region', type=str, default='tail', choices=['full', 'head', 'tail', 'center', 'band'])
    parser.add_argument('--size', type=int, default=512, help='Region size or band half-width')
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--cmap', type=str, default=None)
    parser.add_argument('--annot', action='store_true', help='Annotate values on heatmap when region is small (<=64)')
    parser.add_argument('--log', action='store_true', help='Use log scale (log10) to enhance contrast')
    parser.add_argument('--clip_pct', type=float, default=0.0, help='Symmetric percentile clip (e.g., 1.0)')
    args = parser.parse_args()

    attn = load_attention(args.attn_dir, args.layer, args.head)
    sel = select_region(attn, args.region, args.size)

    # Plot attention heatmap
    region_tag = f"{args.region}{args.size}" if args.region != 'full' else 'full'
    attn_out = os.path.join(args.out_dir, f"attn_L{args.layer}_H{args.head}_{region_tag}.png")
    plot_attention(sel, attn_out, title=f"Layer {args.layer} Head {args.head} ({args.region}, {args.size})", cmap=args.cmap, dpi=args.dpi, annot=args.annot, log=args.log, clip_pct=args.clip_pct)

    # Plot per-position Neff curve for this layer (optionally highlight tail window)
    neff = load_neff(args.attn_dir, args.layer)
    if neff is not None and neff.shape[0] > args.head:
        neff_vec = neff[args.head]
        neff_out = os.path.join(args.out_dir, f"neff_L{args.layer}_H{args.head}.png")
        plot_neff_curve(neff_vec, neff_out, title=f"Neff per position (L{args.layer} H{args.head})", dpi=160)

    print(f"Saved: {attn_out}")
    if neff is not None:
        print(f"Saved: {neff_out}")


if __name__ == '__main__':
    main() 