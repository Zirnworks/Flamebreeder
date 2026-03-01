<script lang="ts">
  import { store } from "../stores/genomes.svelte";
  import GenomeCard from "./GenomeCard.svelte";

  let entries = $derived(store.list);

  function handleSelect(id: string) {
    store.toggleSelect(id);
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
