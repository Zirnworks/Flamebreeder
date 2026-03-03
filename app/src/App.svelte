<script lang="ts">
  import { onMount } from "svelte";
  import { checkServerHealth } from "./lib/api";
  import { store } from "./lib/stores/genomes.svelte";
  import Gallery from "./lib/components/Gallery.svelte";
  import BreedingPanel from "./lib/components/BreedingPanel.svelte";
  import InterpolationSlider from "./lib/components/InterpolationSlider.svelte";
  import GeneEditor from "./lib/components/GeneEditor.svelte";
  import TimeformPanel from "./lib/components/TimeformPanel.svelte";
  import TimeformViewer from "./lib/components/TimeformViewer.svelte";
  import SplitPane from "./lib/components/SplitPane.svelte";

  let activeTab: "breed" | "interpolate" | "edit" | "favorites" | "timeform" = $state("breed");
  let timeformPreviewImages: string[] = $state([]);
  let serverReady = $state(false);
  let serverError = $state("");
  let loadingStatus = $state("Loading model...");

  onMount(() => {
    pollHealth();
  });

  async function pollHealth() {
    for (let i = 0; i < 60; i++) {
      try {
        await checkServerHealth();
        loadingStatus = "Loading saved favorites...";
        const count = await store.loadFromServer();
        if (count > 0) loadingStatus = `Loaded ${count} favorites`;
        serverReady = true;
        return;
      } catch {
        await new Promise((r) => setTimeout(r, 1000));
      }
    }
    serverError = "Could not connect to inference server after 60 seconds.";
  }
</script>

{#if !serverReady}
  <div class="loading-screen">
    {#if serverError}
      <p class="error">{serverError}</p>
      <p class="hint">Make sure the Python server is running:</p>
      <code>python -m breeding.server ~/Data/Praeceptor/models/stylegan2-ada-fractals-k30-kimg880.pkl</code>
    {:else}
      <div class="spinner"></div>
      <p>{loadingStatus}</p>
      <p class="hint">Starting inference server on MPS</p>
    {/if}
  </div>
{:else}
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
          class:active={activeTab === "edit"}
          onclick={() => (activeTab = "edit")}
        >
          Edit Genes
        </button>
        <button
          class:active={activeTab === "interpolate"}
          onclick={() => (activeTab = "interpolate")}
        >
          Interpolate
        </button>
        <button
          class:active={activeTab === "timeform"}
          onclick={() => (activeTab = "timeform")}
        >
          Timeform {store.timeformKeyframes.length > 0 ? `(${store.timeformKeyframes.length})` : ""}
        </button>
        <button
          class:active={activeTab === "favorites"}
          onclick={() => (activeTab = "favorites")}
        >
          Favorites {store.favorites.length > 0 ? `(${store.favorites.length})` : ""}
        </button>
      </nav>
    </header>

    <main>
      {#if activeTab === "edit"}
        <GeneEditor />
      {:else if activeTab === "timeform"}
        <SplitPane initialSplit={40}>
          {#snippet left()}
            <Gallery />
          {/snippet}
          {#snippet right()}
            <TimeformViewer images={timeformPreviewImages} />
          {/snippet}
        </SplitPane>
      {:else if activeTab === "favorites"}
        <Gallery filter="favorites" />
      {:else}
        <Gallery />
      {/if}
    </main>

    <footer>
      {#if activeTab === "breed"}
        <BreedingPanel />
      {:else if activeTab === "interpolate"}
        <InterpolationSlider />
      {:else if activeTab === "timeform"}
        <TimeformPanel onPreview={(imgs) => (timeformPreviewImages = imgs)} />
      {/if}
    </footer>
  </div>
{/if}

<style>
  .loading-screen {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    gap: 16px;
    color: var(--text-secondary);
  }

  .loading-screen p {
    font-size: 16px;
  }

  .loading-screen .hint {
    font-size: 13px;
    opacity: 0.6;
  }

  .loading-screen .error {
    color: var(--danger);
    font-weight: 600;
  }

  .loading-screen code {
    font-size: 12px;
    padding: 8px 16px;
    background: var(--bg-tertiary);
    border-radius: var(--radius);
    color: var(--text-primary);
  }

  .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

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
