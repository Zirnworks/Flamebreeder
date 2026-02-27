"""CLIP-based clustering and redundancy analysis for the training dataset.

Subcommands:
    embed       Extract CLIP embeddings for all images
    cluster     Run k-means clustering and generate inspection grids
    redundancy  Analyze within-cluster similarity and flag redundant images
    export      Export cluster labels in NVIDIA dataset.json format
"""

import json
import os
from pathlib import Path

import click
import numpy as np
from PIL import Image
from tqdm import tqdm


@click.group()
def cli():
    """CLIP clustering and redundancy analysis for fractal flame dataset."""
    pass


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--data-dir", "-d", required=True, help="Directory of training PNGs.")
@click.option("--output", "-o", required=True, help="Output directory for embeddings.")
@click.option("--model", "-m", default="ViT-B-32", help="CLIP model name.")
@click.option("--batch-size", "-b", default=64, help="Batch size for encoding.")
def embed(data_dir: str, output: str, model: str, batch_size: int):
    """Extract CLIP embeddings for all images."""
    import open_clip
    import torch

    data_dir = Path(data_dir).expanduser().resolve()
    out_dir = Path(output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect PNGs
    filenames = sorted(f.name for f in data_dir.iterdir()
                       if f.suffix == ".png" and f.is_file())
    click.echo(f"Found {len(filenames)} images in {data_dir}")

    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        click.echo("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        click.echo("Using CUDA")
    else:
        device = torch.device("cpu")
        click.echo("Using CPU")

    # Load CLIP model
    click.echo(f"Loading CLIP model {model}...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model, pretrained="openai", device=device,
    )
    clip_model.eval()

    # Extract embeddings in batches
    all_embeddings = []
    for i in tqdm(range(0, len(filenames), batch_size), desc="Embedding"):
        batch_names = filenames[i:i + batch_size]
        batch_images = []
        for name in batch_names:
            img = Image.open(data_dir / name).convert("RGB")
            batch_images.append(preprocess(img))

        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad(), torch.amp.autocast(device_type=str(device)):
            features = clip_model.encode_image(batch_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().float().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    click.echo(f"Embeddings shape: {embeddings.shape}")

    # Save
    np.save(out_dir / "embeddings.npy", embeddings)
    with open(out_dir / "filenames.json", "w") as f:
        json.dump(filenames, f)

    click.echo(f"Saved embeddings to {out_dir / 'embeddings.npy'}")
    click.echo(f"Saved filenames to {out_dir / 'filenames.json'}")


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--analysis-dir", "-a", required=True, help="Directory with embeddings.")
@click.option("--data-dir", "-d", required=True, help="Directory of training PNGs.")
@click.option("--k", required=True, type=int, help="Number of clusters.")
@click.option("--grid-size", default=6, help="Grid dimensions for preview (grid_size x grid_size).")
@click.option("--seed", default=42, help="Random seed for k-means.")
def cluster(analysis_dir: str, data_dir: str, k: int, grid_size: int, seed: int):
    """Run k-means clustering and generate inspection grids."""
    from sklearn.cluster import KMeans

    analysis_dir = Path(analysis_dir).expanduser().resolve()
    data_dir = Path(data_dir).expanduser().resolve()

    # Load embeddings
    embeddings = np.load(analysis_dir / "embeddings.npy")
    with open(analysis_dir / "filenames.json") as f:
        filenames = json.load(f)

    click.echo(f"Loaded {len(filenames)} embeddings of dim {embeddings.shape[1]}")
    click.echo(f"Running k-means with k={k}...")

    km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
    labels = km.fit_predict(embeddings)
    centroids = km.cluster_centers_

    # Build cluster info
    clusters = []
    for cid in range(k):
        mask = labels == cid
        member_indices = np.where(mask)[0]
        member_names = [filenames[i] for i in member_indices]
        count = len(member_names)

        # Cosine similarity to centroid for each member
        centroid = centroids[cid]
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
        member_embeds = embeddings[member_indices]
        sims = member_embeds @ centroid_norm
        sorted_idx = np.argsort(-sims)

        # Representatives: closest to centroid
        n_reps = min(grid_size * grid_size, count)
        rep_indices = sorted_idx[:n_reps]
        representatives = [member_names[i] for i in rep_indices]

        # Intra-cluster density: mean pairwise cosine similarity (sample if large)
        if count <= 500:
            pairwise = member_embeds @ member_embeds.T
            np.fill_diagonal(pairwise, 0)
            density = float(pairwise.sum() / max(count * (count - 1), 1))
        else:
            sample_idx = np.random.RandomState(seed).choice(count, 500, replace=False)
            sample = member_embeds[sample_idx]
            pairwise = sample @ sample.T
            np.fill_diagonal(pairwise, 0)
            density = float(pairwise.sum() / (500 * 499))

        clusters.append({
            "id": cid,
            "count": count,
            "density": round(density, 4),
            "representatives": representatives,
            "members": member_names,
        })

    # Save clusters.json
    clusters_data = {"k": k, "seed": seed, "clusters": clusters}
    with open(analysis_dir / "clusters.json", "w") as f:
        json.dump(clusters_data, f, indent=2)

    # Save labels array for redundancy subcommand
    np.save(analysis_dir / "labels.npy", labels)

    # Generate inspection grids
    grids_dir = analysis_dir / "cluster_grids"
    if grids_dir.exists():
        for f in grids_dir.glob("*.png"):
            f.unlink()
    grids_dir.mkdir(exist_ok=True)

    click.echo("Generating inspection grids...")
    thumb_size = 128
    for cluster_info in tqdm(clusters, desc="Grids"):
        cid = cluster_info["id"]
        count = cluster_info["count"]
        reps = cluster_info["representatives"]

        n_cols = grid_size
        n_rows = min(grid_size, (len(reps) + n_cols - 1) // n_cols)
        grid_img = Image.new("RGB",
                             (n_cols * thumb_size, n_rows * thumb_size),
                             (0, 0, 0))

        for idx, name in enumerate(reps[:n_rows * n_cols]):
            row, col = divmod(idx, n_cols)
            try:
                img = Image.open(data_dir / name).convert("RGB")
                img = img.resize((thumb_size, thumb_size), Image.LANCZOS)
                grid_img.paste(img, (col * thumb_size, row * thumb_size))
            except Exception:
                pass

        grid_name = f"cluster_{cid:02d}_n{count}.png"
        grid_img.save(grids_dir / grid_name)

    # Print summary table
    click.echo(f"\n{'ID':>4}  {'Count':>7}  {'Density':>8}")
    click.echo("-" * 24)
    for c in sorted(clusters, key=lambda x: -x["count"]):
        click.echo(f"{c['id']:>4}  {c['count']:>7,}  {c['density']:>8.4f}")

    click.echo(f"\nTotal: {len(filenames)} images in {k} clusters")
    click.echo(f"Grids saved to {grids_dir}")


# ---------------------------------------------------------------------------
# redundancy
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--analysis-dir", "-a", required=True, help="Directory with embeddings and clusters.")
@click.option("--data-dir", "-d", required=True, help="Directory of training PNGs.")
@click.option("--threshold", "-t", default=0.95, help="Cosine similarity threshold for flagging.")
@click.option("--target-max", default=0, help="Max images per cluster (0 = no cap).")
def redundancy(analysis_dir: str, data_dir: str, threshold: float, target_max: int):
    """Analyze within-cluster similarity and flag redundant images."""
    analysis_dir = Path(analysis_dir).expanduser().resolve()
    data_dir = Path(data_dir).expanduser().resolve()

    embeddings = np.load(analysis_dir / "embeddings.npy")
    labels = np.load(analysis_dir / "labels.npy")
    with open(analysis_dir / "filenames.json") as f:
        filenames = json.load(f)
    with open(analysis_dir / "clusters.json") as f:
        clusters_data = json.load(f)

    k = clusters_data["k"]
    centroids_from_clusters = {}

    # Recompute centroids from embeddings for distance calculations
    for cid in range(k):
        mask = labels == cid
        centroids_from_clusters[cid] = embeddings[mask].mean(axis=0)

    all_flagged = set()
    per_cluster_report = []

    click.echo(f"Analyzing redundancy (threshold={threshold}, target_max={target_max})...")

    for cid in tqdm(range(k), desc="Clusters"):
        mask = labels == cid
        member_indices = np.where(mask)[0]
        member_embeds = embeddings[member_indices]
        count = len(member_indices)

        centroid = centroids_from_clusters[cid]
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
        dists_to_centroid = member_embeds @ centroid_norm  # higher = closer

        flagged_sim = set()
        flagged_cap = set()

        # Similarity-based flagging
        if count > 1:
            # For large clusters, process in blocks to avoid memory issues
            if count > 5000:
                # Sample-based approach for very large clusters
                sim_matrix = np.zeros((count, count), dtype=np.float32)
                block_size = 1000
                for i in range(0, count, block_size):
                    end_i = min(i + block_size, count)
                    sim_matrix[i:end_i] = member_embeds[i:end_i] @ member_embeds.T
            else:
                sim_matrix = member_embeds @ member_embeds.T

            # Find pairs above threshold
            already_flagged = set()
            # Process from furthest-from-centroid to closest (flag the outliers first)
            order = np.argsort(dists_to_centroid)  # ascending = furthest first
            for local_idx in order:
                if local_idx in already_flagged:
                    continue
                # Find all images too similar to this one
                similar = np.where(sim_matrix[local_idx] > threshold)[0]
                similar = [s for s in similar if s != local_idx and s not in already_flagged]
                if similar:
                    # Keep whichever is closest to centroid, flag the rest
                    group = [local_idx] + list(similar)
                    group_dists = [(g, dists_to_centroid[g]) for g in group]
                    group_dists.sort(key=lambda x: -x[1])  # closest first
                    keeper = group_dists[0][0]
                    for g, _ in group_dists[1:]:
                        already_flagged.add(g)
                        flagged_sim.add(int(member_indices[g]))

        # Cap-based flagging
        if target_max > 0 and count > target_max:
            # Keep the target_max closest to centroid, flag the rest
            order_desc = np.argsort(-dists_to_centroid)
            for local_idx in order_desc[target_max:]:
                global_idx = int(member_indices[local_idx])
                if global_idx not in flagged_sim:
                    flagged_cap.add(global_idx)

        all_flagged.update(flagged_sim)
        all_flagged.update(flagged_cap)

        per_cluster_report.append({
            "id": cid,
            "count": count,
            "flagged": len(flagged_sim) + len(flagged_cap),
            "kept": count - len(flagged_sim) - len(flagged_cap),
            "reason_breakdown": {
                "similarity": len(flagged_sim),
                "oversized": len(flagged_cap),
            },
        })

    # Build flagged file list
    flagged_files = sorted(filenames[i] for i in all_flagged)

    # Save report
    report = {
        "threshold": threshold,
        "target_max": target_max,
        "total_images": len(filenames),
        "total_flagged": len(flagged_files),
        "projected_remaining": len(filenames) - len(flagged_files),
        "per_cluster": per_cluster_report,
        "flagged_files": flagged_files,
    }
    with open(analysis_dir / "redundancy_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Create symlinks directory for easy review
    symlink_dir = analysis_dir / "redundant_images"
    if symlink_dir.exists():
        for f in symlink_dir.iterdir():
            f.unlink()
    else:
        symlink_dir.mkdir()

    for name in flagged_files:
        src = data_dir / name
        dst = symlink_dir / name
        if src.exists():
            os.symlink(src, dst)

    # Print summary
    click.echo(f"\n{'ID':>4}  {'Count':>7}  {'Flagged':>8}  {'Kept':>7}  {'Sim':>5}  {'Cap':>5}")
    click.echo("-" * 44)
    for c in sorted(per_cluster_report, key=lambda x: -x["flagged"]):
        if c["flagged"] > 0:
            click.echo(
                f"{c['id']:>4}  {c['count']:>7,}  {c['flagged']:>8,}  "
                f"{c['kept']:>7,}  "
                f"{c['reason_breakdown']['similarity']:>5}  "
                f"{c['reason_breakdown']['oversized']:>5}"
            )

    click.echo(f"\nTotal flagged: {len(flagged_files)} / {len(filenames)}")
    click.echo(f"Projected remaining: {len(filenames) - len(flagged_files)}")
    click.echo(f"Report: {analysis_dir / 'redundancy_report.json'}")
    click.echo(f"Symlinks: {symlink_dir}")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--analysis-dir", "-a", required=True, help="Directory with clusters.json.")
@click.option("--data-dir", "-d", required=True, help="Directory of training PNGs.")
@click.option("--output", "-o", default=None, help="Output path for dataset.json.")
def export(analysis_dir: str, data_dir: str, output: str | None):
    """Export cluster labels in NVIDIA dataset.json format."""
    analysis_dir = Path(analysis_dir).expanduser().resolve()
    data_dir = Path(data_dir).expanduser().resolve()
    output_path = Path(output).expanduser().resolve() if output else data_dir / "dataset.json"

    with open(analysis_dir / "clusters.json") as f:
        clusters_data = json.load(f)

    # Build filename -> cluster_id mapping
    file_to_cluster = {}
    for cluster_info in clusters_data["clusters"]:
        cid = cluster_info["id"]
        for name in cluster_info["members"]:
            file_to_cluster[name] = cid

    # Get current images in data_dir (may have been culled)
    current_files = sorted(f.name for f in data_dir.iterdir()
                           if f.suffix == ".png" and f.is_file())

    # Build labels list
    labels = []
    unassigned = []
    for name in current_files:
        if name in file_to_cluster:
            labels.append([name, file_to_cluster[name]])
        else:
            unassigned.append(name)

    if unassigned:
        click.echo(f"WARNING: {len(unassigned)} images have no cluster assignment")
        click.echo("  (These were likely added after clustering. Re-run embed + cluster.)")

    # Write NVIDIA format
    dataset = {"labels": labels}
    with open(output_path, "w") as f:
        json.dump(dataset, f)

    # Write placeholder cluster names
    k = clusters_data["k"]
    names = {str(i): f"cluster_{i:02d}" for i in range(k)}
    names_path = analysis_dir / "cluster_names.json"
    with open(names_path, "w") as f:
        json.dump(names, f, indent=2)

    click.echo(f"Exported {len(labels)} labels to {output_path}")
    click.echo(f"  Clusters: {k}")
    if unassigned:
        click.echo(f"  Unassigned: {len(unassigned)}")
    click.echo(f"  Placeholder names: {names_path}")


# ---------------------------------------------------------------------------

def main():
    cli()


if __name__ == "__main__":
    main()
