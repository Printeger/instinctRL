# M7-014 Known Limitations

- Qualification covers exactly one scenario, one seed, two environments and eight prescribed policy rows. It is not a campaign or statistical study.
- Only `open_field` is READY/executable. The other nine Handbook scenario rows remain NOT_READY; `dynamic_obstacle` remains the sole optional row.
- The scene qualification is plane-only. The full-stage oracle distinguishes the ground subtree, exact ray-scene plane and TASLAB articulation geometry from forbidden scenario geometry; it does not qualify obstacle, terrain, wall, tunnel, clutter, doorway, dynamic actor or randomization behavior.
- Reset and termination are intentionally absent in the bounded run; scenario reset/termination behavior is not qualified here.
- B0 anchor and ICS remain disabled by their accepted baseline contract. Learned policies and B1-B8 were not executed as scenarios.
- Reported metric values are wiring evidence only. No threshold, ranking, flight-quality, safety or comparative-performance conclusion is made.
- Independent Torch archives have different serialization hashes while their complete decoded contents compare exactly.
- The 25-entry live inventory count is specific to the accepted two-environment runtime topology. Any extra geometry/collision prim fails closed rather than being treated as a supported topology variation.
