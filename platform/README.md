# Industrial AI SOP Platform

`platform/` is the clean root of the new product.

The old `sop_system/` and `SOP前后端/` directories are legacy candidates, not
the foundation of this directory. New product code must not import them
directly. A useful legacy capability may only enter through a documented
adapter under `services/runtime/adapters/legacy/`.

## Planned Layout

```text
platform/
  apps/
    web/                  Product UI
    api/                  Business API and application services
  services/
    runtime/              SOP state, rules, evidence and execution
    worker/               Camera capture and inference workers
  packages/
    contracts/            Versioned API and event contracts
    domain/               Shared domain types without infrastructure code
  infra/
    db/                   Database migrations and seed data
    deploy/               Local and server deployment definitions
  tests/
    contract/             Cross-module contract tests
    e2e/                  Product workflow tests
  docs/                   Architecture and product decisions
```

Directories will be created when their first real feature is implemented.
Empty frameworks and placeholder pages are intentionally avoided.

## Development Rules

1. Build by complete product capability, not by scattered pages.
2. Keep dependency direction one-way: UI -> API -> application -> domain.
3. Runtime and workers communicate through versioned contracts.
4. Domain code must not depend on HTTP, database, camera SDK or model runtime.
5. Legacy integration is replaceable and cannot leak legacy data structures.
6. Every running process has one owner, one port and one documented command.
7. Configuration comes from typed config and environment variables, not magic constants.
8. A feature is incomplete without error handling, tests and observable status.
9. Avoid premature microservices; split a service only for a measured reason.
10. No generated files, model weights, recordings or runtime logs in Git.

See [Architecture](docs/ARCHITECTURE.md) for module boundaries and acceptance
rules. See [Domain Model](docs/DOMAIN_MODEL.md) for industrial workstation,
operator action and judgement semantics. See
[Vision/Runtime Boundary](docs/VISION_RUNTIME_BOUNDARY.md) for algorithm
responsibilities.

External research decisions and the real-workstation validation strategy are
recorded in [Research Decisions](docs/RESEARCH_DECISIONS_20260612.md) and
[Evaluation Plan](docs/EVALUATION_PLAN.md).
Fast operator actions are handled according to
[Fast Action Strategy](docs/FAST_ACTION_STRATEGY.md).
The required browser product surface is defined in
[Product UI](docs/PRODUCT_UI.md).

## Current Verification

Run the domain and runtime tests with:

```powershell
.\platform\scripts\test.ps1
```
