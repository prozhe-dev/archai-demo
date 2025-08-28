import { useFloorplan } from "@/hooks/use-floorplan";
import { zIndex } from "@/utils";
import { useTexture } from "@react-three/drei";
import { useMemo } from "react";
import * as THREE from "three";
import { animated, useSpring } from "@react-spring/three";

interface FloorPlanImageProps {
  minimap?: boolean;
}
export default function FloorPlanImage({ minimap = false }: FloorPlanImageProps) {
  const { data, camera, scene, setCamera, floorplanImage } = useFloorplan();
  if (!data || data.canvas.length < 4 || !floorplanImage) return null;

  const texture = useTexture(floorplanImage);

  // Extract size from canvas points
  const width = data.canvas[2][0] - data.canvas[0][0];
  const height = data.canvas[1][1] - data.canvas[0][1];

  const { animatedOpacity } = useSpring({
    animatedOpacity: minimap ? (camera.mode === "bev" && camera.state === "idle" ? 0 : 1) : camera.mode === "bev" && camera.state === "idle" ? 0.7 : 0,
    config: { tension: 300, friction: 20 },
  });

  function toggleCamera() {
    if (!scene?.camera || camera.state === "transitioning") return;

    setCamera({
      ...camera,
      state: "transitioning",
      mode: camera.mode === "pov" ? "bev" : "pov",
      startTime: Date.now(),
      ...(camera.mode === "pov" && {
        pov: {
          position: scene.camera.position.clone(),
          rotation: scene.camera.rotation.clone(),
        },
      }),
    });
  }

  return (
    <mesh
      onClick={toggleCamera}
      position={[width / 2, 2 + zIndex(3), height / 2]} // Center the plane
      rotation={[-Math.PI / 2, 0, 0]} // Rotate to lie flat
    >
      <planeGeometry args={[width, height]} />
      <animated.meshBasicMaterial map={texture} blendColor="white" depthWrite={false} side={THREE.FrontSide} transparent opacity={animatedOpacity} />
    </mesh>
  );
}
