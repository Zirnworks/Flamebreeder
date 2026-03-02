<script lang="ts">
  import { CLUSTERS, NUM_CLUSTERS, normalizeLabel, oneHotLabel, uniformLabel } from "../data/clusters";

  let {
    value = $bindable(uniformLabel()),
    compact = false,
    onchange,
  }: {
    value: number[];
    compact?: boolean;
    onchange?: (label: number[]) => void;
  } = $props();

  let expanded = $state(true);
  let locked = $state(new Array(NUM_CLUSTERS).fill(false));
  // Initialize from compact prop (only captured once, which is fine —
  // compact is a static layout hint, not a reactive toggle)
  $effect(() => { expanded = !compact; });

  let lockCount = $derived(locked.filter(Boolean).length);

  /** Normalize so all values sum to 1, but leave locked sliders untouched. */
  function normalizeWithLocks(label: number[]): number[] {
    const lockedSum = label.reduce((s, v, i) => s + (locked[i] ? v : 0), 0);
    const unlocked = label.map((v, i) => (locked[i] ? 0 : v));
    const unlockedSum = unlocked.reduce((a, b) => a + b, 0);
    const target = Math.max(0, 1 - lockedSum);

    if (unlockedSum <= 0) {
      // All unlocked are zero — distribute evenly among unlocked
      const unlockCount = NUM_CLUSTERS - lockCount;
      if (unlockCount === 0) return label;
      return label.map((v, i) => (locked[i] ? v : target / unlockCount));
    }

    const scale = target / unlockedSum;
    return label.map((v, i) => (locked[i] ? v : v * scale));
  }

  function setSlider(idx: number, val: number) {
    value[idx] = val;
    value = normalizeWithLocks([...value]);
    onchange?.(value);
  }

  function toggleLock(idx: number) {
    locked[idx] = !locked[idx];
    locked = [...locked];
  }

  function selectPure(idx: number) {
    value = oneHotLabel(idx);
    locked = new Array(NUM_CLUSTERS).fill(false);
    onchange?.(value);
  }

  function randomize() {
    const raw = Array.from({ length: NUM_CLUSTERS }, () => Math.random());
    // Sparsify: zero out most clusters for more distinctive results
    const topK = 3;
    const sorted = raw.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v);
    const sparse = new Array(NUM_CLUSTERS).fill(0);
    for (let k = 0; k < topK; k++) {
      sparse[sorted[k].i] = sorted[k].v;
    }
    // Keep locked sliders at current values
    for (let i = 0; i < NUM_CLUSTERS; i++) {
      if (locked[i]) sparse[i] = value[i];
    }
    value = normalizeWithLocks(sparse);
    onchange?.(value);
  }

  // Active clusters (weight > 1%) for the blend indicator
  let activeSliders = $derived(
    value
      .map((w, i) => ({ weight: w, cluster: CLUSTERS[i] }))
      .filter((s) => s.weight > 0.01)
      .sort((a, b) => b.weight - a.weight),
  );
</script>

<div class="cluster-sliders" class:compact>
  <!-- Blend indicator bar -->
  <div class="blend-bar">
    {#each activeSliders as s}
      <div
        class="blend-segment"
        style="flex: {s.weight}; background: {s.cluster.color};"
        title="{s.cluster.label}: {(s.weight * 100).toFixed(0)}%"
      ></div>
    {/each}
  </div>

  <div class="controls-row">
    <button class="small" onclick={randomize}>Randomize</button>
    {#if lockCount > 0}
      <button class="small" onclick={() => { locked = new Array(NUM_CLUSTERS).fill(false); }}>
        Clear Locks ({lockCount})
      </button>
    {/if}
    {#if compact}
      <button class="small" onclick={() => (expanded = !expanded)}>
        {expanded ? "Collapse" : "Expand"} Sliders
      </button>
    {/if}
  </div>

  {#if expanded}
    <div class="sliders">
      {#each CLUSTERS as cluster (cluster.id)}
        {@const weight = value[cluster.id] ?? 0}
        <div class="slider-row" class:active={weight > 0.01} class:locked={locked[cluster.id]}>
          <button
            class="lock-btn"
            class:is-locked={locked[cluster.id]}
            title={locked[cluster.id] ? "Unlock" : "Lock"}
            onclick={() => toggleLock(cluster.id)}
          >
            {locked[cluster.id] ? "\u{1F512}" : "\u{1F513}"}
          </button>
          <button
            class="cluster-btn"
            style="border-left: 3px solid {cluster.color};"
            title={cluster.description}
            onclick={() => selectPure(cluster.id)}
          >
            {cluster.label}
          </button>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={weight}
            oninput={(e) => setSlider(cluster.id, parseFloat(e.currentTarget.value))}
          />
          <span class="weight">{(weight * 100).toFixed(0)}%</span>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .cluster-sliders {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .blend-bar {
    display: flex;
    height: 6px;
    border-radius: 3px;
    overflow: hidden;
    background: var(--bg-tertiary);
  }

  .blend-segment {
    transition: flex 0.2s ease;
    min-width: 2px;
  }

  .controls-row {
    display: flex;
    gap: 8px;
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

  .sliders {
    display: flex;
    flex-direction: column;
    gap: 2px;
    max-height: 400px;
    overflow-y: auto;
    padding-right: 4px;
  }

  .slider-row {
    display: flex;
    align-items: center;
    gap: 8px;
    opacity: 0.5;
    transition: opacity 0.15s;
  }

  .slider-row.active {
    opacity: 1;
  }

  .slider-row:hover {
    opacity: 1;
  }

  .slider-row.locked {
    opacity: 1;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 4px;
    padding: 1px 4px;
    margin: -1px -4px;
  }

  .lock-btn {
    flex-shrink: 0;
    width: 20px;
    height: 20px;
    padding: 0;
    background: transparent;
    border: none;
    cursor: pointer;
    font-size: 11px;
    opacity: 0.3;
    transition: opacity 0.15s;
    line-height: 1;
  }

  .lock-btn:hover {
    opacity: 0.7;
  }

  .lock-btn.is-locked {
    opacity: 0.9;
  }

  .cluster-btn {
    flex-shrink: 0;
    width: 130px;
    text-align: left;
    font-size: 11px;
    padding: 2px 8px;
    background: transparent;
    color: var(--text-secondary);
    border-radius: 2px;
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .cluster-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  input[type="range"] {
    flex: 1;
    height: 4px;
    accent-color: var(--accent);
  }

  .weight {
    flex-shrink: 0;
    width: 32px;
    text-align: right;
    font-size: 10px;
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
  }

  .compact .sliders {
    max-height: 240px;
  }
</style>
