<script lang="ts">
  import { genomes, selectedParents } from "../stores/genomes";
  import GenomeCard from "./GenomeCard.svelte";

  let entries = $derived.by(() => {
    let map = new Map();
    genomes.subscribe((v) => (map = v))();
    return Array.from(map.values());
  });

  function handleSelect(id: string) {
    selectedParents.update(([a, b]) => {
      if (a === id) return [null, b];
      if (b === id) return [a, null];
      if (a === null) return [id, b];
      if (b === null) return [a, id];
      return [b, id];
    });
  }
</script>

<div class="gallery scrollable">
  {#if entries.length === 0}
    <div class="empty">
      <p>No images yet. Generate some to get started.</p>
    </div>
  {:else}
    <div class="grid">
      {#each entries as entry (entry.id)}
        <GenomeCard {entry} onSelect={handleSelect} />
      {/each}
    </div>
  {/if}
</div>

<style>
  .gallery {
    flex: 1;
    padding: 16px;
  }

  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 12px;
  }

  .empty {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 300px;
    color: var(--text-secondary);
    font-size: 16px;
  }
</style>
