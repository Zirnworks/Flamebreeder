<script lang="ts">
  import { genomes, selectedParents } from "../stores/genomes";
  import { breed, generateRandom } from "../api";
  import type { GenomeEntry } from "../stores/genomes";

  let breedingMethod = $state("average");
  let offspringCount = $state(4);
  let ratio = $state(0.5);
  let truncationPsi = $state(0.7);
  let loading = $state(false);
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

  let canBreed = $derived(parents.a !== null && parents.b !== null && !loading);

  async function handleGenerate() {
    loading = true;
    error = "";
    try {
      const results = await generateRandom(4, truncationPsi);
      genomes.addFromResponse(results);
    } catch (e) {
      error = String(e);
    }
    loading = false;
  }

  async function handleBreed() {
    if (!parents.a || !parents.b) return;
    loading = true;
    error = "";
    try {
      const params: Record<string, unknown> =
        breedingMethod === "average"
          ? { ratio, truncation_psi: truncationPsi }
          : { truncation_psi: truncationPsi };
      const results = await breed(
        parents.a.id,
        parents.b.id,
        breedingMethod,
        params,
        offspringCount,
      );
      genomes.addFromResponse(results);
    } catch (e) {
      error = String(e);
    }
    loading = false;
  }
</script>

<div class="panel">
  <div class="controls">
    <button class="primary" onclick={handleGenerate} disabled={loading}>
      {loading ? "Generating..." : "Generate Random"}
    </button>

    <div class="breed-section">
      <div class="parent-slots">
        <div class="parent-slot" class:filled={parents.a}>
          {#if parents.a}
            <img src={parents.a.imageDataUrl} alt="Parent A" />
          {:else}
            <span>Parent A</span>
          {/if}
        </div>

        <span class="cross">&#215;</span>

        <div class="parent-slot" class:filled={parents.b}>
          {#if parents.b}
            <img src={parents.b.imageDataUrl} alt="Parent B" />
          {:else}
            <span>Parent B</span>
          {/if}
        </div>
      </div>

      <div class="breed-options">
        <label>
          Method
          <select bind:value={breedingMethod}>
            <option value="average">Average (Slerp)</option>
            <option value="crossover">Uniform Crossover</option>
            <option value="block_crossover">Block Crossover</option>
            <option value="guided">Guided</option>
            <option value="style_mix">Style Mix</option>
          </select>
        </label>

        {#if breedingMethod === "average"}
          <label>
            Ratio: {ratio.toFixed(2)}
            <input type="range" min="0" max="1" step="0.05" bind:value={ratio} />
          </label>
        {/if}

        <label>
          Truncation: {truncationPsi.toFixed(2)}
          <input type="range" min="0" max="1" step="0.05" bind:value={truncationPsi} />
        </label>

        <label>
          Offspring
          <input
            type="number"
            min="1"
            max="8"
            bind:value={offspringCount}
            style="width: 60px; background: var(--bg-tertiary); color: var(--text-primary); border: 1px solid var(--border); border-radius: var(--radius); padding: 4px 8px;"
          />
        </label>
      </div>

      <button class="primary" onclick={handleBreed} disabled={!canBreed}>
        {loading ? "Breeding..." : "Breed"}
      </button>
    </div>
  </div>

  {#if error}
    <div class="error">{error}</div>
  {/if}
</div>

<style>
  .panel {
    background: var(--bg-secondary);
    border-top: 1px solid var(--border);
    padding: 16px;
  }

  .controls {
    display: flex;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
  }

  .breed-section {
    display: flex;
    align-items: center;
    gap: 16px;
    flex: 1;
  }

  .parent-slots {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .parent-slot {
    width: 64px;
    height: 64px;
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    color: var(--text-secondary);
    overflow: hidden;
  }

  .parent-slot.filled {
    border-style: solid;
    border-color: var(--accent);
  }

  .parent-slot img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .cross {
    font-size: 24px;
    color: var(--text-secondary);
  }

  .breed-options {
    display: flex;
    gap: 12px;
    align-items: center;
  }

  .breed-options label {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 12px;
    color: var(--text-secondary);
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
