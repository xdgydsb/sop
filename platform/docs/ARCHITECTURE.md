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

The business center is industrial operator work compliance: a worker performs
an operation at a workstation for a work order, and the product determines
whether the observed state transition follows the released SOP. Object
detection is supporting evidence, not the business result.

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

- `ProductionLine`
- `Workstation`
- `Operator`
- `ProductVariant`
- `WorkOrder`
- `AlgorithmVersion`
- `ModelVersion`
- `SopDefinition`
- `SopRelease`
- `OperationDefinition`
- `WorkCycle`
- `OperationAttempt`
- `Judgement`
- `Deviation`
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

The browser interface is a required product surface, not an optional wrapper.
After the execution primitives are established, each new capability must ship
as a vertical slice with its API, runtime behavior, UI states and browser
verification. Detailed UI requirements are defined in `docs/PRODUCT_UI.md`.

## 7. First Vertical Slice

The first slice is **run one real operator work cycle at one workstation**:

1. Register one algorithm/model version.
2. Define each operation's precondition, action, postcondition and error rules.
3. Publish an immutable SOP release for one product variant.
4. Bind one real workstation, camera and compute node.
5. Start a work cycle manually or from a work-order trigger.
6. Show live video and immediate operation judgement.
7. Store corresponding evidence and deviations for every attempted operation.
8. Complete, reject or send the product cycle to manual review.
9. Recover clearly from camera or inference disconnection without blaming the operator.

The existing five-step package SOP may supply adapters for this slice only if it
passes the legacy admission gate.
