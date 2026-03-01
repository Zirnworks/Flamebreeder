<script lang="ts">
  import { store } from "../stores/genomes.svelte";
  import { remapGenome } from "../api";
  import { uniformLabel } from "../data/clusters";
  import ClusterSliders from "./ClusterSliders.svelte";

  let loading = $state(false);
  let error = $state("");
  let truncationPsi = $state(0.7);
  let classLabel = $state(uniformLabel());
  let previewUrl = $state("");

  let target = $derived(
    store.selectedA ? store.get(store.selectedA) ?? null : null,
  );

  let canEdit = $derived(target !== null && target.genome.seed_z !== null);

  // When target changes, load its class label
  $effect(() => {
    if (target?.genome.class_label) {
      classLabel = [...target.genome.class_label];
    } else {
      classLabel = uniformLabel();
    }
    if (target?.genome.truncation_psi != null) {
      truncationPsi = target.genome.truncation_psi;
    }
    previewUrl = target?.imageDataUrl ?? "";
  });

  async function handleRemap() {
    if (!target || !canEdit) return;
    loading = true;
    error = "";
    try {
      const result = await remapGenome(target.id, classLabel, truncationPsi);
      store.addSingle(result);
      previewUrl = `data:image/png;base64,${result.image_base64}`;
    } catch (e) {
      error = String(e);
    }
    loading = false;
  }
</script>

<div class="gene-editor">
  {#if !target}
    <div class="empty">
      <p>Select an image from the gallery to edit its genes.</p>
    </div>
  {:else}
    <div class="editor-layout">
      <div class="preview-section">
        <div class="image-frame">
          {#if previewUrl}
            <img src={previewUrl} alt="Preview" />
          {/if}
          {#if loading}
            <div class="loading-overlay">Remapping...</div>
          {/if}
        </div>
        {#if !canEdit}
          <p class="hint">This image was bred — no seed z-vector available for gene editing. Only generated images can be remapped.</p>
        {/if}
      </div>

      <div class="controls-section">
        <h3>Cluster Genes</h3>
        <ClusterSliders bind:value={classLabel} />

        <div class="psi-control">
          <label>
            Truncation: {truncationPsi.toFixed(2)}
            <input type="range" min="0" max="1" step="0.05" bind:value={truncationPsi} />
          </label>
        </div>

        <button class="primary remap-btn" onclick={handleRemap} disabled={!canEdit || loading}>
          {loading ? "Remapping..." : "Remap with New Genes"}
        </button>

        {#if error}
          <div class="error">{error}</div>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .gene-editor {
    padding: 16px;
    height: 100%;
  }

  .empty {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 200px;
    color: var(--text-secondary);
  }

  .editor-layout {
    display: flex;
    gap: 24px;
    height: 100%;
  }

  .preview-section {
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .image-frame {
    position: relative;
    width: 320px;
    height: 320px;
    border-radius: var(--radius-lg);
    overflow: hidden;
    background: var(--bg-tertiary);
  }

  .image-frame img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .loading-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.6);
    color: white;
    font-size: 14px;
    font-weight: 600;
  }

  .hint {
    font-size: 12px;
    color: var(--text-secondary);
    max-width: 320px;
    line-height: 1.4;
  }

  .controls-section {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 12px;
    overflow-y: auto;
  }

  h3 {
    font-size: 16px;
    font-weight: 600;
    margin: 0;
  }

  .psi-control label {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .remap-btn {
    align-self: flex-start;
  }

  .error {
    padding: 8px 12px;
    background: rgba(204, 68, 68, 0.15);
    border: 1px solid var(--danger);
    border-radius: var(--radius);
    color: var(--danger);
    font-size: 13px;
  }
</style>
