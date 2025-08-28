import { useThree } from "@react-three/fiber";
import { useEffect } from "react";
import { useGenRender } from "@/hooks/use-gen-render";
import * as THREE from "three";

export default function SnapshotTaker() {
  const { gl, scene, camera } = useThree();
  const { isGenRenderModalOpen, setSnapshotImage, setSnapshotDepthImage, setSnapshotEdgesImage, setSnapshotSegmentsImage } = useGenRender();

  function takeSnapshot() {
    // Store original gl state
    const originalViewport = gl.getViewport(new THREE.Vector4());
    const originalScissor = gl.getScissor(new THREE.Vector4());
    const originalScissorTest = gl.getScissorTest();
    const originalSize = gl.getSize(new THREE.Vector2());
    const originalPixelRatio = gl.getPixelRatio();
    const originalLayers = 0;
    const originalEnvironment = scene.environment;
    const originalBackground = scene.background;

    // Get the actual canvas element and its display size
    const canvas = gl.domElement;
    const displayWidth = canvas.clientWidth;
    const displayHeight = canvas.clientHeight;

    // Set gl to full canvas size for the snapshot
    gl.setPixelRatio(1); // Use 1:1 pixel ratio for consistent results
    gl.setSize(displayWidth, displayHeight, false);
    gl.setViewport(0, 0, displayWidth, displayHeight);
    gl.setScissor(0, 0, displayWidth, displayHeight);
    gl.setScissorTest(false);

    // Calculate crop dimensions for 3:2 aspect ratio
    const targetAspectRatio = 3 / 2;
    let cropWidth, cropHeight, offsetX, offsetY;

    if (displayWidth / displayHeight > targetAspectRatio) {
      // Original is wider than 3:2, crop width
      cropHeight = displayHeight;
      cropWidth = displayHeight * targetAspectRatio;
      offsetX = (displayWidth - cropWidth) / 2;
      offsetY = 0;
    } else {
      // Original is taller than 3:2, crop height
      cropWidth = displayWidth;
      cropHeight = displayWidth / targetAspectRatio;
      offsetX = 0;
      offsetY = (displayHeight - cropHeight) / 2;
    }

    // Function to crop and return image data
    const cropAndReturnImage = () => {
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = cropWidth;
      tempCanvas.height = cropHeight;

      const ctx = tempCanvas.getContext("2d")!;
      ctx.drawImage(
        canvas,
        offsetX,
        offsetY,
        cropWidth,
        cropHeight, // Source rectangle
        0,
        0,
        cropWidth,
        cropHeight, // Destination rectangle
      );

      return tempCanvas.toDataURL("image/png");
    };

    // 1. Normal snapshot (layer 0 - default)
    camera.layers.set(0);
    gl.render(scene, camera);
    const normalSnapshot = cropAndReturnImage();

    // Set environment and background to black for non-normal render passes
    scene.environment = null;
    scene.background = new THREE.Color(0x000000);

    // 2. Depth map (layer 1)
    camera.layers.set(1);
    gl.render(scene, camera);
    const depthSnapshot = cropAndReturnImage();

    // 3. Edge map (layer 2)
    camera.layers.set(2);
    gl.render(scene, camera);
    const edgeSnapshot = cropAndReturnImage();

    // 4. Segment map (layer 3)
    camera.layers.set(3);
    gl.render(scene, camera);
    const segmentSnapshot = cropAndReturnImage();

    // Restore original gl state
    gl.setViewport(originalViewport.x, originalViewport.y, originalViewport.z, originalViewport.w);
    gl.setScissor(originalScissor.x, originalScissor.y, originalScissor.z, originalScissor.w);
    gl.setScissorTest(originalScissorTest);
    gl.setSize(originalSize.x, originalSize.y, false);
    gl.setPixelRatio(originalPixelRatio);
    camera.layers.set(originalLayers);

    // Restore original environment and background
    scene.environment = originalEnvironment;
    scene.background = originalBackground;

    return {
      normal: normalSnapshot,
      depth: depthSnapshot,
      edge: edgeSnapshot,
      segment: segmentSnapshot,
    };
  }

  useEffect(() => {
    if (isGenRenderModalOpen) {
      const snapshots = takeSnapshot();
      setSnapshotImage(snapshots.normal);
      setSnapshotDepthImage(snapshots.depth);
      setSnapshotEdgesImage(snapshots.edge);
      setSnapshotSegmentsImage(snapshots.segment);
    }
  }, [isGenRenderModalOpen, gl, scene, camera, setSnapshotImage, setSnapshotDepthImage, setSnapshotEdgesImage, setSnapshotSegmentsImage]);

  return null; // This component doesn't render anything
}
