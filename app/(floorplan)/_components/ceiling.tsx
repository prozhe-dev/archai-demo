import { useFloorplan } from "@/hooks/use-floorplan";
import { zIndex } from "@/utils";
import { useMemo } from "react";
import * as THREE from "three";
import { animated, useSpring } from "@react-spring/three";
import { SEGMENT_COLORS } from "@/utils/consts";
import { Edges } from "@react-three/drei";
import { DepthMaterial } from "@/components/materials/depth-material";

export default function Ceiling() {
  const { data, camera } = useFloorplan();

  const processedGeometry = useMemo(() => {
    if (!data) return null;
    const shape = new THREE.Shape();
    if (data.floor.length > 0) {
      data.floor.forEach((vert) => {
        shape.moveTo(vert[0], vert[1]);
        shape.lineTo(vert[0], vert[1]);
      });
    }
    shape.closePath();
    return new THREE.ShapeGeometry(shape);
  }, [data]);

  const { animatedOpacity } = useSpring({
    animatedOpacity: camera.mode === "pov" && camera.state === "idle" ? 1 : 0,
    config: { tension: 300, friction: 20 },
  });

  if (!data || data.floor.length === 0 || !processedGeometry) return null;

  return (
    <>
      {/* Normal render (layer 0) */}
      <mesh geometry={processedGeometry} position={[0, 2 + zIndex(2), 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[1, -1, 1]}>
        <animated.meshBasicMaterial color="#adadad" side={THREE.DoubleSide} transparent opacity={animatedOpacity} />
      </mesh>

      {/* Depth map (layer 1) */}
      <mesh layers={[1]} geometry={processedGeometry} position={[0, 2 + zIndex(2), 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[1, -1, 1]}>
        {/* <meshDepthMaterial opacity={0.01} depthTest depthWrite side={THREE.DoubleSide} /> */}
        <DepthMaterial side={THREE.DoubleSide} />
      </mesh>

      {/* Edge map (layer 2) */}
      <mesh layers={[2]} geometry={processedGeometry} position={[0, 2 + zIndex(2), 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[1, -1, 1]}>
        <meshBasicMaterial color="black" side={THREE.DoubleSide} />
        <Edges linewidth={3} color="white" threshold={45} layers={[2]} />
      </mesh>

      {/* Segment map (layer 3) */}
      <mesh layers={[3]} geometry={processedGeometry} position={[0, 2 + zIndex(2), 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[1, -1, 1]}>
        <meshBasicMaterial color={SEGMENT_COLORS.ceiling} side={THREE.DoubleSide} />
        <Edges linewidth={2} color="black" threshold={45} layers={[3]} />
      </mesh>
    </>
  );
}
