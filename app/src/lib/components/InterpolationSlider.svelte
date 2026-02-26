<script lang="ts">
  import { genomes, selectedParents } from "../stores/genomes";
  import { interpolate } from "../api";
  import type { GenomeEntry, GenomeWithImage } from "../stores/genomes";

  let steps = $state(10);
  let loading = $state(false);
  let strip: { imageDataUrl: string; t: number }[] = $state([]);
  let error = $state("");

  let parents = $derived.by(() => {
    let sel: [string | null, string | null] = [null, null];
    selectedParents.subscribe((v) => (sel = v))();

    let map = new Map<string, GenomeEntry>();
    genomes.subscribe((v) => (map = v))();

    return {
      a: sel[0] ? map.get(sel[0]) ?? null : null,
      b: sel[1] ? map.get(sel[1]) ?? null : null,
    };
  });

  let canInterpolate = $derived(parents.a !== null && parents.b !== null && !loading);

  async function handleInterpolate() {
    if (!parents.a || !parents.b) return;
    loading = true;
    error = "";
    strip = [];
    try {
      const results = await interpolate(parents.a.id, parents.b.id, steps);
      strip = results.map((r: any, i: number) => ({
        imageDataUrl: `data:image/png;base64,${r.image_base64}`,
        t: i / (steps - 1),
      }));
      genomes.addFromResponse(results);
    } catch (e) {
      error = String(e);
    }
    loading = false;
  }
</script>

<div class="interpolation">
  <div class="header">
    <h3>Interpolation</h3>
    <label>
      Steps
      <input
        type="number"
        min="2"
        max="64"
        bind:value={steps}
        style="width: 60px; background: var(--bg-tertiary); color: var(--text-primary); border: 1px solid var(--border); border-radius: var(--radius); padding: 4px 8px;"
      />
    </label>
    <button class="primary" onclick={handleInterpolate} disabled={!canInterpolate}>
      {loading ? "Generating..." : "Generate Strip"}
    </button>
  </div>

  {#if strip.length > 0}
    <div class="strip scrollable">
      {#each strip as frame}
        <div class="frame">
          <img src={frame.imageDataUrl} alt="Interpolation frame" />
          <span class="t-label">{frame.t.toFixed(2)}</span>
        </div>
      {/each}
    </div>
  {/if}

  {#if error}
    <div class="error">{error}</div>
  {/if}
</div>

<style>
  .interpolation {
    padding: 16px;
  }

  .header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 12px;
  }

  h3 {
    font-size: 16px;
    font-weight: 600;
  }

  .header label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    color: var(--text-secondary);
  }

  .strip {
    display: flex;
    gap: 8px;
    overflow-x: auto;
    padding-bottom: 8px;
  }

  .frame {
    flex-shrink: 0;
    position: relative;
    width: 128px;
    border-radius: var(--radius);
    overflow: hidden;
  }

  .frame img {
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    display: block;
  }

  .t-label {
    position: absolute;
    bottom: 4px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(0, 0, 0, 0.7);
    color: var(--text-secondary);
    padding: 1px 6px;
    border-radius: 8px;
    font-size: 10px;
  }

  .error {
    margin-top: 8px;
    padding: 8px 12px;
    background: rgba(204, 68, 68, 0.15);
    border: 1px solid var(--danger);
    border-radius: var(--radius);
    color: var(--danger);
    font-size: 13px;
  }
</style>
