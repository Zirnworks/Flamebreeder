<script lang="ts">
  import { selectedParents } from "../stores/genomes";
  import { updateGenome } from "../api";
  import type { GenomeEntry } from "../stores/genomes";

  let { entry, onSelect }: { entry: GenomeEntry; onSelect?: (id: string) => void } = $props();

  let isSelected = $derived.by(() => {
    let parents: [string | null, string | null] = [null, null];
    selectedParents.subscribe((v) => (parents = v))();
    return parents[0] === entry.id || parents[1] === entry.id;
  });

  async function toggleFavorite() {
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
    <div class="generation-badge">G{entry.genome.generation}</div>
    {#if entry.genome.favorite}
      <div class="favorite-badge">&#9733;</div>
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

  .generation-badge {
    position: absolute;
    top: 6px;
    left: 6px;
    background: rgba(0, 0, 0, 0.7);
    color: var(--text-secondary);
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
  }

  .favorite-badge {
    position: absolute;
    top: 6px;
    right: 6px;
    color: var(--gold);
    font-size: 18px;
    text-shadow: 0 0 4px rgba(0, 0, 0, 0.8);
  }
</style>
