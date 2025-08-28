import { useFloorplan } from "@/hooks/use-floorplan";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Edges, useTexture } from "@react-three/drei";
import { CuboidCollider, RigidBody, useRapier } from "@react-three/rapier";
import { PLAYER_POS_Y, SEGMENT_COLORS } from "@/utils/consts";
import { useHotkeys } from "react-hotkeys-hook";
import { TriplanarMaterial } from "@/components/materials/triplanar-material";
import { DepthMaterial } from "@/components/materials/depth-material";

export default function Floor() {
  const { data } = useFloorplan();

  if (!data || data.floor.length === 0) return null;

  const geometry = useMemo(() => {
    const shape = new THREE.Shape();
    if (data.floor.length > 0) {
      data.floor.forEach((vert) => {
        shape.moveTo(vert[0], vert[1]);
        shape.lineTo(vert[0], vert[1]);
      });
    }
    shape.closePath();

    return new THREE.ExtrudeGeometry(shape, { depth: 0.2, bevelEnabled: false });
  }, [data]);

  const texture = useTexture("/textures/wood.jpg");
  texture.flipY = false;
  texture.rotation = Math.PI / 2;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;

  return (
    <>
      <RigidBody type="fixed" colliders="cuboid">
        <CuboidCollider args={[1000, 2, 1000]} position={[0, -1, 0]} />
      </RigidBody>

      {/* Normal render (layer 0) */}
      <mesh geometry={geometry} position={[0, 1, 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[1, -1, -1]}>
        <meshBasicMaterial color="#adadad" side={THREE.DoubleSide} />
        {/* <meshStandardMaterial map={texture} side={THREE.DoubleSide} /> */}
        <Edges linewidth={2} color="black" />
      </mesh>

      {/* Depth map (layer 1) */}
      <mesh layers={[1]} geometry={geometry} position={[0, 1, 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[1, -1, -1]}>
        {/* <meshDepthMaterial opacity={0.01} depthTest depthWrite /> */}
        <DepthMaterial />
      </mesh>

      {/* Edge map (layer 2) */}
      <mesh layers={[2]} geometry={geometry} position={[0, 1, 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[1, -1, -1]}>
        <meshBasicMaterial color="black" />
        <Edges linewidth={3} color="white" threshold={45} layers={[2]} />
      </mesh>

      {/* Segment map (layer 3) */}
      <mesh layers={[3]} geometry={geometry} position={[0, 1, 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[1, -1, -1]}>
        <meshBasicMaterial color={SEGMENT_COLORS.floor} />
        <Edges linewidth={2} color="black" threshold={45} layers={[3]} />
      </mesh>
    </>
  );
}
