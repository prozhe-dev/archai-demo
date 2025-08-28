import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Edges, Extrude } from "@react-three/drei";
import { useFloorplan } from "@/hooks/use-floorplan";
import { RapierRigidBody, RigidBody } from "@react-three/rapier";
import { ThreeEvent, useFrame } from "@react-three/fiber";
import { animated, useSpring } from "@react-spring/three";
import { zIndex } from "@/utils";
import { SEGMENT_COLORS } from "@/utils/consts";
import { DepthMaterial } from "@/components/materials/depth-material";

export default function Doors() {
  const { data } = useFloorplan();
  if (!data || data.doors.length === 0) return null;

  const shapes = useMemo(
    () =>
      data.doors.map((door) => {
        const shape = new THREE.Shape();

        // Move to first vertex
        shape.moveTo(door[0][0][0], door[0][0][1]);

        // Draw lines to other vertices
        for (let i = 1; i < door.length; i++) {
          shape.lineTo(door[i][0][0], door[i][0][1]);
        }

        // Close the shape
        shape.closePath();

        return shape;
      }),
    [data],
  );

  return shapes.map((shape) => <Door key={shape.uuid} shape={shape} />);
}

function Door({ shape }: { shape: THREE.Shape }) {
  const rigidRef = useRef<RapierRigidBody>(null);

  const [isOpen, setIsOpen] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [currentAngle, setCurrentAngle] = useState(0);

  const targetAngle = isOpen ? -Math.PI / 2.2 : 0;

  const geometry = useMemo(() => new THREE.ExtrudeGeometry(shape, { depth: 1, bevelEnabled: false }), [shape]);

  const { pivotX, pivotZ } = useMemo(() => {
    geometry.computeBoundingBox();
    const boundingBox = geometry.boundingBox!;
    return {
      pivotX: boundingBox.max.x,
      pivotZ: boundingBox.min.y,
    };
  }, [geometry]);

  useFrame((_, delta) => {
    if (!rigidRef.current) return;
    const speed = 2;
    const diff = targetAngle - currentAngle;
    if (Math.abs(diff) > 0.001) {
      const step = Math.sign(diff) * Math.min(speed * delta, Math.abs(diff));
      const newAngle = currentAngle + step;
      setCurrentAngle(newAngle);

      const quat = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, newAngle, 0));
      rigidRef.current.setNextKinematicRotation(quat);
    }
  });

  const { color } = useSpring({
    // color: isHovered ? "white" : "white",
    color: isHovered ? "white" : "#462C2C",
    config: { tension: 300, friction: 20 },
  });

  const handleClick = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    setIsOpen((prev) => !prev);
  };

  return (
    <RigidBody ref={rigidRef} type="kinematicPosition" colliders="trimesh" position={[pivotX, 0, pivotZ]}>
      <group position={[-pivotX, 0, -pivotZ]} onClick={handleClick} onPointerEnter={() => setIsHovered(true)} onPointerLeave={() => setIsHovered(false)}>
        {/* Normal render (layer 0) */}
        <animated.mesh geometry={geometry} position={[0, 2 + zIndex(1), 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[1, -1, -1]}>
          <animated.meshBasicMaterial color={color} />
          <Edges geometry={geometry} linewidth={2} color="black" threshold={45} />
        </animated.mesh>

        {/* Depth map (layer 1) */}
        <mesh layers={[1]} geometry={geometry} position={[0, 2 + zIndex(1), 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[1, -1, -1]}>
          {/* <meshDepthMaterial opacity={0.01} depthTest depthWrite /> */}
          <DepthMaterial />
        </mesh>

        {/* Edge map (layer 2) */}
        <mesh layers={[2]} geometry={geometry} position={[0, 2 + zIndex(1), 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[1, -1, -1]}>
          <meshBasicMaterial color="black" />
          <Edges geometry={geometry} linewidth={3} color="white" threshold={45} layers={[2]} />
        </mesh>

        {/* Segment map (layer 3) */}
        <mesh layers={[3]} geometry={geometry} position={[0, 2 + zIndex(1), 0]} rotation={[-Math.PI / 2, 0, 0]} scale={[1, -1, -1]}>
          <meshBasicMaterial color={SEGMENT_COLORS.door} />
          <Edges geometry={geometry} linewidth={2} color="black" threshold={45} layers={[3]} />
        </mesh>
      </group>
    </RigidBody>
  );
}
