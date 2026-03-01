<script lang="ts">
  import { store } from "../stores/genomes.svelte";
  import { updateGenome } from "../api";
  import { dominantCluster } from "../data/clusters";
  import type { GenomeEntry } from "../stores/genomes.svelte";

  let { entry, onSelect }: { entry: GenomeEntry; onSelect?: (id: string) => void } = $props();

  let isSelected = $derived(
    store.selectedA === entry.id || store.selectedB === entry.id,
  );

  let cluster = $derived(dominantCluster(entry.genome.class_label));

  async function toggleFavorite(e: Event) {
    e.stopPropagation();
    const newFav = !entry.genome.favorite;
    await updateGenome(entry.id, undefined, newFav);
  }

  function handleClick() {
    onSelect?.(entry.id);
  }
</script>

<div
  class="card"
  class:selected={isSelected}
  role="button"
  tabindex="0"
  onclick={handleClick}
  onkeydown={(e) => e.key === "Enter" && handleClick()}
>
  <div class="image-container">
    <img src={entry.imageDataUrl} alt="Fractal flame" />
    <div class="badges-top">
      <div class="generation-badge">G{entry.genome.generation}</div>
      {#if entry.genome.favorite}
        <button class="favorite-badge active" onclick={toggleFavorite}>&#9733;</button>
      {:else}
        <button class="favorite-badge" onclick={toggleFavorite}>&#9734;</button>
      {/if}
    </div>
    {#if cluster}
      <div class="cluster-tag" style="border-color: {cluster.color};">
        {cluster.label}
      </div>
    {/if}
  </div>
</div>

<style>
  .card {
    position: relative;
    border-radius: var(--radius-lg);
    overflow: hidden;
    border: 2px solid transparent;
    transition: all 0.2s ease;
    cursor: pointer;
  }

  .card:hover {
    border-color: var(--border);
    transform: scale(1.02);
  }

  .card.selected {
    border-color: var(--accent);
    box-shadow: 0 0 16px var(--accent-glow);
  }

  .image-container {
    position: relative;
    aspect-ratio: 1;
  }

  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }

  .badges-top {
    position: absolute;
    top: 6px;
    left: 6px;
    right: 6px;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }

  .generation-badge {
    background: rgba(0, 0, 0, 0.7);
    color: var(--text-secondary);
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
  }

  .favorite-badge {
    background: rgba(0, 0, 0, 0.5);
    border: none;
    color: var(--text-secondary);
    font-size: 16px;
    cursor: pointer;
    padding: 2px 4px;
    border-radius: 10px;
    line-height: 1;
  }

  .favorite-badge.active {
    color: var(--gold);
  }

  .favorite-badge:hover {
    color: var(--gold);
    background: rgba(0, 0, 0, 0.7);
  }

  .cluster-tag {
    position: absolute;
    bottom: 6px;
    left: 6px;
    background: rgba(0, 0, 0, 0.75);
    color: var(--text-secondary);
    padding: 2px 8px;
    border-radius: 8px;
    font-size: 10px;
    border-left: 2px solid;
    max-width: calc(100% - 16px);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
</style>
