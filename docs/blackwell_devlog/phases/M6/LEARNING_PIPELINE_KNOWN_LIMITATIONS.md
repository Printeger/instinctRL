# M6-004 Bounded Learning-Pipeline Known Limitations

- This is a fixed migration-correctness adapter and smoke, not a general
  collector or production training CLI.
- The strict checkpoint restores actor, critic, both optimizers, update counter,
  and RNG. It does not save or restore environment/simulator state; segment two
  continues in the same live environment.
- Qualification is limited to two environments, eight policy steps, 160
  physics steps, 16 samples, two updates, and one forced automatic timeout per
  independent process.
- The timeout counter adjustment is test-fixture setup only. It is not product
  curriculum, randomization, termination logic, or training behavior.
- Runtime qualification covers one RTX 5070 under the frozen R580 envelope. It
  does not establish throughput, scaling, convergence, reward quality,
  tracking, or flight-performance claims.
- General logging, evaluation, export, ROS/deployment packaging, recurrent
  state, replay, asynchronous/distributed collection, and simulator restart
  are not implemented.
- Peak CUDA allocation was 143,420,928 bytes in each formal pipeline process.
