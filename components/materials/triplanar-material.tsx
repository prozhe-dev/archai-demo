import { shaderMaterial } from "@react-three/drei";
import { extend, ThreeElement } from "@react-three/fiber";
import * as THREE from "three";

const TriplanarMaterialImpl = shaderMaterial(
  {
    uTexture: null as THREE.Texture | null,
    uTextureNormal: null as THREE.Texture | null,
    uScale: 0.5,
  },
  /* glsl */ `
    varying vec3 vP;
    varying vec3 vN;
    varying vec2 vUv;
    uniform highp sampler2D uTextureNormal;
    varying vec3 vWorldPos;
    varying vec3 vWorldSpaceNormal;

    void main() {
      vP = position;
      vN = normal;
      vUv = uv;
      vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
      vWorldSpaceNormal = (modelMatrix * vec4(vN, 0.0)).xyz;
    }
  `,
  /* glsl */ `
    uniform highp sampler2D uTexture;
    uniform highp sampler2D uTextureNormal;
    uniform highp float uScale;
    varying vec3 vP;
    varying vec3 vN;
    varying vec2 vUv;
    varying vec3 vWorldPos;
    varying vec3 vWorldSpaceNormal;

    mat2 rotate(float angle) {
        float s = sin(angle);
        float c = cos(angle);
        return mat2(c, -s, s, c);
    }

    vec3 triplanarTexture(sampler2D uTexture,vec3 normal,vec3 pos) {
        vec3 whichAxisIsMax = vec3(0.0);
        vec3 normalizedNormal = normalize(normal);
        vec3 sfumata = vec3(pow(abs(normalizedNormal.x),4.0),pow(abs(normalizedNormal.y),4.0),pow(abs(normalizedNormal.z),4.0));
        // Determina quale asse ha il valore più elevato, forse non serve
        //if (sfumata.x >= sfumata.y && sfumata.x >= sfumata.z) {
        //  whichAxisIsMax.x = 1.0;
        //} else if (normal.y >= normal.x && normal.y >= normal.z) {
        // whichAxisIsMax.y = 1.0;
        //} else {
      //   whichAxisIsMax.z = 1.0;
        //}
        vec3 xColor = texture2D(uTexture, pos.zy*uScale).rgb;
        vec3 yColor = texture2D(uTexture, pos.xz*uScale).rgb;
        vec3 zColor = texture2D(uTexture, pos.xy*uScale).rgb;
        sfumata = vec3(xColor * sfumata.x + yColor * sfumata.y  + zColor * sfumata.z);
        return vec3(sfumata);
    }

    vec4 _CalculateLighting(
        vec3 lightDirection, vec3 lightColour, vec3 worldSpaceNormal, vec3 viewDirection) {
      float diffuse = saturate(dot(worldSpaceNormal, lightDirection));

      vec3 H = normalize(lightDirection + viewDirection);
      float NdotH = dot(worldSpaceNormal, H);
      float specular = saturate(pow(NdotH, 8.0));

      return vec4(lightColour * (diffuse + diffuse * specular), 0);
    }


    vec4 _ComputeLighting(vec3 worldSpaceNormal, vec3 sunDir, vec3 viewDirection) {
      // Hardcoded, whee!
      vec4 lighting;
      
      lighting += _CalculateLighting(
          sunDir, vec3(1.25, 1.25, 1.25), worldSpaceNormal, viewDirection);
      lighting += _CalculateLighting(
          -sunDir, vec3(0.75, 0.75, 1.0), worldSpaceNormal, viewDirection);
      lighting += _CalculateLighting(
          vec3(0, 1, 0), vec3(0.25, 0.25, 0.25), worldSpaceNormal, viewDirection);
      
      return lighting;
    }

    vec3 triplanarNormal(vec3 pos,vec3 normal,float texSlice,sampler2D tex) {
    vec3 tx = texture2D(tex, pos.zy*uScale).rgb*vec3(2.0,2.0,2.0)-vec3(1.0,1.0,1.0);
    vec3 ty = texture2D(tex, pos.xz*uScale).rgb*vec3(2.0,2.0,2.0)-vec3(1.0,1.0,1.0);
    vec3 tz = texture2D(tex, pos.xy*uScale).rgb*vec3(2.0,2.0,2.0)-vec3(1.0,1.0,1.0);
    vec3 weights = abs(normal.xyz);
    weights *= weights;
    weights = weights/ (weights.x + weights.y + weights.z);
    vec3 axis = sign(normal);

    vec3 tangentX = normalize(cross(normal, vec3(0.0,axis.x,0.0)));
    vec3 bitangentX = normalize(cross(tangentX, normal))* axis.x;
    mat3 tbnX = mat3(tangentX,bitangentX,normal);
    
    vec3 tangentY = normalize(cross(normal, vec3(0.0,0.0,axis.y)));
    vec3 bitangentY = normalize(cross(tangentY, normal))* axis.y;
    mat3 tbnY = mat3(tangentY,bitangentY,normal);

    vec3 tangentZ = normalize(cross(normal, vec3(0.0,-axis.z,0.0)));
    vec3 bitangentZ = normalize(cross(tangentZ, normal))* axis.z;
    mat3 tbnZ = mat3(tangentZ,bitangentZ,normal);

    vec3 worldNormal = normalize(
      clamp(tbnX*tx,-1.0,1.0)*weights.x +
      clamp(tbnY*ty,-1.0,1.0)*weights.y +
      clamp(tbnZ*tz,-1.0,1.0)*weights.z);
      return worldNormal;
    }


    void main() {
      vec3 sunDir = normalize(vec3(-0.5, 1.0, -0.3));
      vec3 eyeDirection = normalize(vWorldPos - cameraPosition);
      vec3 tnormal = triplanarNormal(vP, vN, 0.5, uTextureNormal);
      vec4 lighting = _ComputeLighting(vWorldSpaceNormal, sunDir, eyeDirection);

      //  Calcola il termine di diffusione di Lambert
      float diff = max(dot(tnormal, lighting.xyz), 0.0);

      // Calcoliamo la specularità
      vec3 reflectDir = reflect(-eyeDirection, tnormal);
      float spec = pow(max(dot(eyeDirection, reflectDir), 0.0), 16.0);
      vec3 specular = vec3(0.5) * spec;

      vec3 tcolor = triplanarTexture(uTexture, tnormal,vP);
      vec3 finalColour = tcolor+ mix(vec3(1.0, 1.0, 1.0), tcolor.xyz, 0.5) * tcolor;
      vec3 diffuse = finalColour*(1.0+diff);

      csm_DiffuseColor = vec4(diffuse+specular,1.0);
    }
  `,
);

extend({ TriplanarMaterial: TriplanarMaterialImpl });

export function TriplanarMaterial(props: ThreeElement<typeof TriplanarMaterialImpl>) {
  return <triplanarMaterial {...props} />;
}

declare module "@react-three/fiber" {
  interface ThreeElements {
    triplanarMaterial: ThreeElement<typeof TriplanarMaterialImpl>;
  }
}
