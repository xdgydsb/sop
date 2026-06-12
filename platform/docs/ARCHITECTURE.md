# Product Architecture

## 1. Architecture Style

Start with a modular product rather than a collection of scripts:

- One web application.
- One business API using a modular-monolith structure.
- One SOP runtime service.
- One or more camera/inference workers.
- Versioned contracts between control-plane and execution-plane components.

This keeps deployment understandable while allowing camera and GPU workloads to
run independently. Components may be split later only when deployment,
performance or team ownership requires it.

## 2. Product Modules

### Control Plane

- Dataset and annotation management.
- Algorithm and model registry.
- SOP visual designer and version publishing.
- Camera, compute node and deployment management.
- Users, permissions and audit records.

### Execution Plane

- Camera acquisition and stream health.
- Model inference adapters.
- Temporal event generation.
- SOP rule and state execution.
- Evidence screenshots and action clips.
- Worker heartbeat, metrics and failure recovery.

### Operations Plane

- Live monitoring.
- Task history and evidence review.
- Alarm handling.
- Quality, cycle-time and device-health statistics.

## 3. Mandatory Dependency Direction

```text
web
  -> API contracts
api application
  -> domain
  -> repository/runtime ports
infrastructure adapters
  -> ports

worker
  -> execution contracts
  -> camera/model adapters

legacy code
  -> legacy adapter
  -> normalized execution contracts
```

Forbidden dependencies:

- Web pages directly reading databases or worker files.
- Domain logic importing framework, ORM, camera SDK or model code.
- Runtime logic depending on UI state.
- New modules importing files from legacy directories.
- Model-specific labels leaking into generic workflow code.

## 4. Core Contracts

The first implementation must define these versioned contracts before building
pages:

- `AlgorithmVersion`
- `ModelVersion`
- `SopDefinition`
- `SopRelease`
- `CameraSource`
- `ComputeNode`
- `Deployment`
- `RuntimeSession`
- `DetectionObservation`
- `SopEvent`
- `StepEvidence`
- `Alarm`

All timestamps, IDs, status values and error responses must have one canonical
definition.

## 5. Legacy Admission Gate

A legacy component is classified as one of:

- `REUSE`: correct, measured, testable and cleanly isolated.
- `WRAP`: useful behavior but poor interface; place behind an adapter.
- `REPLACE`: requirements are valid but implementation is not maintainable.
- `DROP`: no product value or duplicates another capability.

No legacy component enters the product until its classification and evidence
are recorded in an architecture decision.

## 6. Definition of Done

A vertical product capability is done only when it includes:

- User-visible workflow.
- API and persisted domain behavior.
- Runtime integration when applicable.
- Loading, empty, failure and recovery states.
- Structured logs and health information.
- Unit or contract tests.
- End-to-end acceptance check.
- Documentation of the single startup path.

A page with static data, a button without backend behavior, or an inference
result without evidence is not a completed feature.

## 7. First Vertical Slice

The first slice is **publish and run one real SOP**:

1. Register one algorithm/model version.
2. Create an SOP definition with ordered steps and temporal rules.
3. Bind one real camera and one compute node.
4. Publish an immutable SOP release.
5. Start, stop and reset a runtime session.
6. Show live video and immediate step state.
7. Store corresponding evidence for every accepted event.
8. Recover clearly from camera or inference disconnection.

The existing five-step package SOP may supply adapters for this slice only if it
passes the legacy admission gate.
