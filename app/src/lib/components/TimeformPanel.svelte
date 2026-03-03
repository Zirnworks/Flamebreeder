<script lang="ts">
  import { save } from "@tauri-apps/plugin-dialog";
  import { store } from "../stores/genomes.svelte";
  import { exportTimeform, interpolate } from "../api";

  let {
    onPreview,
  }: {
    onPreview?: (images: string[]) => void;
  } = $props();

  let totalFrames = $state(128);
  let spacing = $state<"uniform" | "adaptive">("uniform");
  let textureSize = $state(256);
  let totalDepth = $state(10.0);
  let quadSize = $state(5.0);
  let method = $state<"slerp" | "lerp">("slerp");
  let showAdvanced = $state(false);
  let loading = $state(false);
  let previewing = $state(false);
  let error = $state("");
  let exportedPath = $state("");

  let keyframes = $derived(
    store.timeformKeyframes
      .map((id) => store.get(id))
      .filter((e): e is NonNullable<typeof e> => e != null),
  );
  let canExport = $derived(keyframes.length >= 2 && !loading && !previewing);

  function addSelected() {
    if (store.selectedA) store.addTimeformKeyframe(store.selectedA);
    if (store.selectedB) store.addTimeformKeyframe(store.selectedB);
  }

  async function handlePreview() {
    if (keyframes.length < 2) return;
    previewing = true;
    error = "";
    try {
      // Distribute the requested frame count across segments
      const previewSteps = Math.max(4, Math.round(totalFrames / (keyframes.length - 1)));
      const allImages: string[] = [];

      for (let i = 0; i < keyframes.length - 1; i++) {
        const results = await interpolate(
          keyframes[i].id,
          keyframes[i + 1].id,
          previewSteps,
          method,
        );
        for (let j = 0; j < results.length; j++) {
          // Skip first frame of subsequent segments (duplicate of prev last)
          if (i > 0 && j === 0) continue;
          allImages.push(`data:image/png;base64,${results[j].image_base64}`);
        }
      }
      onPreview?.(allImages);
    } catch (e) {
      error = String(e);
    }
    previewing = false;
  }

  async function handleExport() {
    const path = await save({
      defaultPath: "timeform.glb",
      filters: [{ name: "GLB", extensions: ["glb"] }],
    });
    if (!path) return;

    loading = true;
    error = "";
    exportedPath = "";
    try {
      await exportTimeform(
        {
          keyframeIds: store.timeformKeyframes,
          totalFrames,
          spacing,
          textureSize,
          totalDepth,
          quadSize,
          interpolationMethod: method,
        },
        path,
      );
      exportedPath = path;
    } catch (e) {
      error = String(e);
    }
    loading = false;
  }
</script>

<div class="timeform-panel">
  <div class="keyframes-section">
    <div class="section-header">
      <h3>Keyframes ({keyframes.length})</h3>
      <div class="section-actions">
        <button class="small" onclick={addSelected} disabled={!store.selectedA}>
          Add Selected
        </button>
        <button
          class="small"
          onclick={() => store.clearTimeformKeyframes()}
          disabled={keyframes.length === 0}
        >
          Clear
        </button>
      </div>
    </div>

    {#if keyframes.length === 0}
      <p class="hint">Select images from the gallery and click "Add Selected" to build your keyframe sequence.</p>
    {:else}
      <div class="keyframe-strip">
        {#each keyframes as kf, i (kf.id)}
          <div class="keyframe-thumb">
            <img src={kf.imageDataUrl} alt="Keyframe {i}" />
            <button
              class="remove-btn"
              onclick={() => store.removeTimeformKeyframe(kf.id)}
            >
              &times;
            </button>
            <span class="keyframe-index">{i + 1}</span>
          </div>
        {/each}
      </div>
    {/if}
  </div>

  <div class="config-section">
    <label>
      Frames: {totalFrames}
      <input type="range" min="16" max="512" step="16" bind:value={totalFrames} />
    </label>

    <label>
      Spacing
      <select bind:value={spacing}>
        <option value="uniform">Uniform</option>
        <option value="adaptive">Adaptive</option>
      </select>
    </label>

    <label>
      Texture
      <select bind:value={textureSize}>
        <option value={64}>64px</option>
        <option value={128}>128px</option>
        <option value={256}>256px</option>
        <option value={512}>512px</option>
      </select>
    </label>

    <button class="small" onclick={() => (showAdvanced = !showAdvanced)}>
      {showAdvanced ? "Hide" : "Show"} Advanced
    </button>

    {#if showAdvanced}
      <label>
        Depth: {totalDepth.toFixed(1)}
        <input type="range" min="1" max="100" step="1" bind:value={totalDepth} />
      </label>
      <label>
        Quad Size: {quadSize.toFixed(1)}
        <input type="range" min="0.5" max="50" step="0.5" bind:value={quadSize} />
      </label>
      <label>
        Interpolation
        <select bind:value={method}>
          <option value="slerp">SLERP</option>
          <option value="lerp">LERP</option>
        </select>
      </label>
    {/if}
  </div>

  <div class="export-section">
    <button class="secondary" onclick={handlePreview} disabled={!canExport}>
      {previewing ? "Generating..." : "Preview 3D"}
    </button>
    <button class="primary" onclick={handleExport} disabled={!canExport}>
      {#if loading}
        Exporting...
      {:else}
        Export GLB
      {/if}
    </button>

    {#if exportedPath}
      <p class="success">Exported to {exportedPath}</p>
    {/if}
    {#if error}
      <p class="error">{error}</p>
    {/if}
  </div>
</div>

<style>
  .timeform-panel {
    padding: 12px 20px;
    display: flex;
    gap: 20px;
    align-items: flex-start;
    border-top: 1px solid var(--border);
    background: var(--bg-secondary);
  }

  .keyframes-section {
    flex: 1;
    min-width: 0;
  }

  .section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
  }

  .section-header h3 {
    font-size: 14px;
    font-weight: 600;
    margin: 0;
  }

  .section-actions {
    display: flex;
    gap: 6px;
  }

  .hint {
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.4;
  }

  .keyframe-strip {
    display: flex;
    gap: 6px;
    overflow-x: auto;
    padding: 4px 0;
  }

  .keyframe-thumb {
    position: relative;
    flex-shrink: 0;
    width: 128px;
    height: 128px;
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--border);
  }

  .keyframe-thumb img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .remove-btn {
    position: absolute;
    top: 2px;
    right: 2px;
    width: 16px;
    height: 16px;
    padding: 0;
    background: rgba(0, 0, 0, 0.7);
    color: var(--text-secondary);
    border: none;
    border-radius: 50%;
    font-size: 12px;
    line-height: 1;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .remove-btn:hover {
    background: var(--danger);
    color: white;
  }

  .keyframe-index {
    position: absolute;
    bottom: 2px;
    left: 2px;
    background: rgba(0, 0, 0, 0.7);
    color: var(--text-secondary);
    font-size: 9px;
    padding: 1px 4px;
    border-radius: 6px;
  }

  .config-section {
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-width: 180px;
  }

  .config-section label {
    display: flex;
    flex-direction: column;
    gap: 3px;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .config-section select {
    font-size: 11px;
    padding: 3px 6px;
  }

  .export-section {
    display: flex;
    flex-direction: column;
    gap: 8px;
    align-items: flex-start;
  }

  .small {
    font-size: 11px;
    padding: 3px 10px;
    background: var(--bg-tertiary);
    color: var(--text-secondary);
    border-radius: 12px;
  }

  .small:hover {
    background: var(--border);
    color: var(--text-primary);
  }

  .success {
    font-size: 11px;
    color: var(--success);
  }

  .error {
    font-size: 11px;
    color: var(--danger);
    max-width: 200px;
  }
</style>
