<script lang="ts">
  import Gallery from "./lib/components/Gallery.svelte";
  import BreedingPanel from "./lib/components/BreedingPanel.svelte";
  import InterpolationSlider from "./lib/components/InterpolationSlider.svelte";

  let activeTab: "breed" | "interpolate" = $state("breed");
</script>

<div class="app">
  <header>
    <h1>Fractal Breeder</h1>
    <nav>
      <button
        class:active={activeTab === "breed"}
        onclick={() => (activeTab = "breed")}
      >
        Breed
      </button>
      <button
        class:active={activeTab === "interpolate"}
        onclick={() => (activeTab = "interpolate")}
      >
        Interpolate
      </button>
    </nav>
  </header>

  <main>
    <Gallery />
  </main>

  <footer>
    {#if activeTab === "breed"}
      <BreedingPanel />
    {:else}
      <InterpolationSlider />
    {/if}
  </footer>
</div>

<style>
  .app {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }

  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
  }

  h1 {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.02em;
  }

  nav {
    display: flex;
    gap: 4px;
  }

  nav button {
    background: transparent;
    color: var(--text-secondary);
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
  }

  nav button:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  nav button.active {
    background: var(--accent);
    color: white;
  }

  main {
    flex: 1;
    overflow-y: auto;
  }

  footer {
    flex-shrink: 0;
  }
</style>
