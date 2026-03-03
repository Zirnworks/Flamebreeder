<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import * as THREE from "three";
  import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

  let { images = [] }: { images: string[] } = $props();

  let container: HTMLDivElement;
  let renderer: THREE.WebGLRenderer | null = null;
  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;
  let controls: OrbitControls;
  let animFrameId: number;
  let planesGroup: THREE.Group;

  function init() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0f);

    camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
    camera.position.set(0, 0, 12);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    planesGroup = new THREE.Group();
    scene.add(planesGroup);

    resize();
    animate();
  }

  function buildPlanes(srcs: string[]) {
    // Clear existing
    while (planesGroup.children.length) {
      const child = planesGroup.children[0] as THREE.Mesh;
      child.geometry.dispose();
      if (child.material instanceof THREE.Material) child.material.dispose();
      planesGroup.remove(child);
    }

    if (srcs.length === 0) return;

    const totalDepth = 6;
    const quadSize = 5;
    const loader = new THREE.TextureLoader();

    srcs.forEach((src, i) => {
      const t = srcs.length > 1 ? i / (srcs.length - 1) : 0;
      const z = (t - 0.5) * totalDepth;

      loader.load(src, (texture) => {
        // Read the texture to create an alpha map from luminance
        const canvas = document.createElement("canvas");
        canvas.width = texture.image.width;
        canvas.height = texture.image.height;
        const ctx = canvas.getContext("2d")!;
        ctx.drawImage(texture.image, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;

        // Apply alpha = max(R, G, B)
        for (let p = 0; p < data.length; p += 4) {
          data[p + 3] = Math.max(data[p], data[p + 1], data[p + 2]);
        }
        ctx.putImageData(imageData, 0, 0);

        const alphaTexture = new THREE.CanvasTexture(canvas);
        alphaTexture.colorSpace = THREE.SRGBColorSpace;

        const material = new THREE.MeshBasicMaterial({
          map: alphaTexture,
          transparent: true,
          side: THREE.DoubleSide,
          depthWrite: false,
        });

        const geometry = new THREE.PlaneGeometry(quadSize, quadSize);
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.z = z;
        planesGroup.add(mesh);
      });
    });
  }

  function resize() {
    if (!renderer || !container) return;
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w === 0 || h === 0) return;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function animate() {
    animFrameId = requestAnimationFrame(animate);
    controls?.update();
    renderer?.render(scene, camera);
  }

  onMount(() => {
    init();
    const ro = new ResizeObserver(() => resize());
    ro.observe(container);
    return () => ro.disconnect();
  });

  onDestroy(() => {
    cancelAnimationFrame(animFrameId);
    renderer?.dispose();
    controls?.dispose();
  });

  // Rebuild planes when images change
  $effect(() => {
    if (renderer) buildPlanes(images);
  });
</script>

<div class="viewer" bind:this={container}>
  {#if images.length === 0}
    <div class="empty-overlay">
      <p>Generate a preview to see the timeform in 3D</p>
    </div>
  {/if}
</div>

<style>
  .viewer {
    position: relative;
    width: 100%;
    height: 100%;
    min-height: 200px;
    background: var(--bg-primary);
    border-radius: var(--radius-lg);
    overflow: hidden;
  }

  .viewer :global(canvas) {
    display: block;
    width: 100% !important;
    height: 100% !important;
  }

  .empty-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-secondary);
    font-size: 13px;
    pointer-events: none;
  }
</style>
