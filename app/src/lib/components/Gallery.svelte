<script lang="ts">
  import { store } from "../stores/genomes.svelte";
  import GenomeCard from "./GenomeCard.svelte";
  import Lightbox from "./Lightbox.svelte";

  let { filter }: { filter?: "favorites" } = $props();

  let entries = $derived(filter === "favorites" ? store.favorites : store.list);
  let lightboxSrc = $state("");

  function handleSelect(id: string) {
    store.toggleSelect(id);
  }

  function handleDblClick(id: string) {
    const entry = store.get(id);
    if (entry) lightboxSrc = entry.imageDataUrl;
  }
</script>

{#if lightboxSrc}
  <Lightbox src={lightboxSrc} onclose={() => (lightboxSrc = "")} />
{/if}

<div class="gallery scrollable">
  {#if entries.length === 0}
    <div class="empty">
      <p>{filter === "favorites" ? "No favorites yet. Star images to save them here." : "No images yet. Generate some to get started."}</p>
    </div>
  {:else}
    <div class="grid">
      {#each entries as entry (entry.id)}
        <GenomeCard {entry} onSelect={handleSelect} onDblClick={handleDblClick} />
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
