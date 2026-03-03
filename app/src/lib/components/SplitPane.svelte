<script lang="ts">
  import type { Snippet } from "svelte";

  let { left, right, initialSplit = 50 }: {
    left: Snippet;
    right: Snippet;
    initialSplit?: number;
  } = $props();

  let splitPercent = $state(initialSplit);
  let dragging = $state(false);
  let pane: HTMLDivElement;

  function onPointerDown(e: PointerEvent) {
    dragging = true;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }

  function onPointerMove(e: PointerEvent) {
    if (!dragging || !pane) return;
    const rect = pane.getBoundingClientRect();
    const x = e.clientX - rect.left;
    splitPercent = Math.max(15, Math.min(85, (x / rect.width) * 100));
  }

  function onPointerUp() {
    dragging = false;
  }
</script>

<div
  class="split-pane"
  bind:this={pane}
  onpointermove={onPointerMove}
  onpointerup={onPointerUp}
>
  <div class="pane left" style="width: {splitPercent}%;">
    {@render left()}
  </div>
  <div
    class="divider"
    class:active={dragging}
    onpointerdown={onPointerDown}
    role="separator"
    aria-valuenow={Math.round(splitPercent)}
    tabindex="0"
  ></div>
  <div class="pane right" style="width: {100 - splitPercent}%;">
    {@render right()}
  </div>
</div>

<style>
  .split-pane {
    display: flex;
    width: 100%;
    height: 100%;
    overflow: hidden;
  }

  .pane {
    overflow: auto;
    min-width: 0;
  }

  .pane.left {
    flex-shrink: 0;
  }

  .pane.right {
    flex: 1;
    min-width: 0;
  }

  .divider {
    flex-shrink: 0;
    width: 6px;
    cursor: col-resize;
    background: var(--border);
    transition: background 0.15s;
    touch-action: none;
  }

  .divider:hover,
  .divider.active {
    background: var(--accent);
  }
</style>
