import React, { useRef } from "react";
import { useGLTF } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { MathUtils } from "three";

export default function FingerModel({ angles = {}, ...props }) {
  const { nodes, materials } = useGLTF("/Finger v9.6.glb");

  const baseRef = useRef();
  const link1Ref = useRef();
  const link2Ref = useRef();
  const link3Ref = useRef();

  useFrame(() => {
    if (baseRef.current) {
      const maxZAngle = MathUtils.degToRad(90);

      baseRef.current.rotation.z = (angles.base || 0) * maxZAngle;
    }

    if (link1Ref.current) link1Ref.current.rotation.x = angles.j1 || 0;
    if (link2Ref.current) link2Ref.current.rotation.x = angles.j2 || 0;
    if (link3Ref.current) link3Ref.current.rotation.x = angles.j3 || 0;
  });

  return (
    <group {...props} dispose={null}>
      <group rotation={[Math.PI / 2, 0, 0]} scale={0.001}>
        <mesh
          castShadow
          receiveShadow
          geometry={nodes.Body1003.geometry}
          material={materials["Steel - Satin"]}
          scale={10}
        />

        <group position={[1, 12.583, 16.652]} rotation={[-2.487, 0, Math.PI]}>
          <group ref={baseRef}>
            <mesh
              castShadow
              receiveShadow
              geometry={nodes.Body4.geometry}
              material={materials["Steel - Satin"]}
              scale={10}
            />

            <group
              position={[10.357, 31.839, 19.198]}
              rotation={[0.436, -0.262, 0]}
            >
              <group ref={link1Ref}>
                <mesh
                  castShadow
                  receiveShadow
                  geometry={nodes.Body1002.geometry}
                  material={materials["Steel - Satin"]}
                  scale={10}
                />

                <group position={[0, 38.5, 0]}>
                  <group ref={link2Ref}>
                    <mesh
                      castShadow
                      receiveShadow
                      geometry={nodes.Body1001.geometry}
                      material={materials["Steel - Satin"]}
                      scale={10}
                    />

                    <group position={[0, 38.5, 0]}>
                      <group ref={link3Ref}>
                        <mesh
                          castShadow
                          receiveShadow
                          geometry={nodes.Body1.geometry}
                          material={materials["Steel - Satin"]}
                          scale={10}
                        />
                      </group>
                    </group>
                  </group>
                </group>
              </group>
            </group>
          </group>
        </group>
      </group>
    </group>
  );
}

useGLTF.preload("/Finger v9.6.glb");
