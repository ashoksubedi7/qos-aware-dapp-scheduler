# AssuredQoS

AssuredQoS is a research framework for QoS-aware deep reinforcement learning scheduling in O-RAN.

The project studies whether formal neural-network verification can uncover QoS-critical scheduling decisions that are difficult to expose through conventional simulation, and whether those formal counterexamples can be used to improve the learned scheduler.

The final target is an O-RAN scheduling dApp whose learned policy is evaluated through conventional network experiments, formal property analysis, counterexample reproduction, and counterexample-guided repair.

## Research Workflow

Training
→ QoS evaluation
→ ONNX export
→ formal verification
→ counterexample discovery
→ simulator reproduction
→ policy repair
→ formal re-verification
→ O-RAN dApp evaluation

## Current Status

Project initialization and simulator instrumentation.
