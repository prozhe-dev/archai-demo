"use client";

import { useFloorplan } from "@/hooks/use-floorplan";
import React, { useRef } from "react";
import { Grid, PerspectiveCamera } from "@react-three/drei";
import { CAMERA_FOV } from "@/utils/consts";
import FloorPlanImage from "./floorplan-image";
import { Group } from "three";
import { useFrame } from "@react-three/fiber";

export default function Minimap() {
  const { debug, camera, bounds, scene } = useFloorplan();
  const playerIndicatorRef = useRef<Group>(null);

  useFrame(() => {
    if (playerIndicatorRef.current && scene.camera?.position && scene.camera?.rotation) {
      playerIndicatorRef.current.position.x = scene.camera.position.x;
      playerIndicatorRef.current.position.z = scene.camera.position.z;
      playerIndicatorRef.current.rotation.z = scene.camera.rotation.y;
      }
  });

  return (
    <>
      <color attach="background" args={["white"]} />
      <PerspectiveCamera makeDefault fov={CAMERA_FOV} position={[camera.bev.position.x, camera.bev.position.y * 0.8, camera.bev.position.z]} rotation={camera.bev.rotation} />

      <group ref={playerIndicatorRef} position={[0, 0.1, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <mesh>
          <circleGeometry args={[0.2, 32]} />
          <meshBasicMaterial color="red" transparent opacity={0.7} />
        </mesh>

        <mesh position={[0, 0.4, 0]}>
          <coneGeometry args={[0.1, 0.2, 3]} />
          <meshBasicMaterial color="red" transparent opacity={0.7} />
        </mesh>
      </group>

      <group position={[-bounds.center.x, -2, -bounds.center.z]}>
        <FloorPlanImage minimap />
      </group>

      {(debug || camera.mode === "bev") && (
        <Grid position={[0, 0, 0]} args={[20, 20]} cellSize={1} cellThickness={0.5} cellColor="#cccccc" sectionSize={5} sectionThickness={1} sectionColor="#999999" infiniteGrid={true} />
      )}
    </>
  );
}
