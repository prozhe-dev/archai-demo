import { shaderMaterial } from "@react-three/drei";
import { extend, ThreeElement } from "@react-three/fiber";

const DepthMaterialImpl = shaderMaterial(
  {
    uBrightness: 0.5,
    uDepthRange: 6.0, // Control the depth range
    uContrast: 1.0, // Control contrast
  },
  /* glsl */ `
    varying float vDepth;
    void main() {
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      vDepth = -mvPosition.z;
      gl_Position = projectionMatrix * mvPosition;
    }
  `,
  /* glsl */ `
    uniform float uBrightness;
    uniform float uDepthRange;
    uniform float uContrast;
    varying float vDepth;
    void main() {
      // Inverted depth mapping (closer objects are brighter)
      float normalizedDepth = 1.0 - clamp(vDepth / uDepthRange, 0.0, 1.0);
      // Apply contrast adjustment
      normalizedDepth = pow(normalizedDepth, uContrast);
      // Apply brightness adjustment - higher values make it brighter
      float brightDepth = normalizedDepth * uBrightness;
      gl_FragColor = vec4(vec3(brightDepth), 1.0);
    }
  `,
);

extend({ DepthMaterial: DepthMaterialImpl });

export function DepthMaterial(props: ThreeElement<typeof DepthMaterialImpl>) {
  return <depthMaterial depthTest depthWrite {...props} />;
}

declare module "@react-three/fiber" {
  interface ThreeElements {
    depthMaterial: ThreeElement<typeof DepthMaterialImpl>;
  }
}
