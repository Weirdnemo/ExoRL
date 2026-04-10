# ExoRL Deployment Roadmap (Next Release)

This roadmap turns the planned deployment upgrades into an execution order that is easy to track in GitHub issues/milestones.

## Proposed Milestone

**Milestone name:** `vNext Deployment & Distribution`

**Goal:** Make ExoRL easy to install, validate cross-platform reliability, and fully automate release delivery (tags -> version bump -> build -> publish).

## Scope (Must-Have for vNext)

### 1) Packaging Extras and Dependency Groups

- [ ] Add extras in project metadata:
  - [ ] `exorl[rl]` for SB3 + PyTorch stack
  - [ ] `exorl[science]` for astropy + matplotlib stack
  - [ ] `exorl[dev]` for lint/test/docs tooling
  - [ ] `exorl[all]` as the union of all runtime extras
- [ ] Verify `pip install -e ".[rl]"`, `".[science]"`, `".[dev]"`, `".[all]"`.
- [ ] Document each extra in `README.md` with one-line intent and examples.

### 2) Docker Distribution

- [ ] Add `Dockerfile` for full training/research image (`exorl/exorl:latest`).
- [ ] Add `Dockerfile.cpu` (or equivalent) for lightweight CI/dev image (`exorl/exorl:cpu-only`).
- [ ] Add image build and push workflow on release/tag.
- [ ] Add quickstart `docker run` examples in docs.

### 3) CI Compatibility Matrix

- [ ] Add Python matrix: `3.9`, `3.10`, `3.11`, `3.12`.
- [ ] Add OS matrix: `ubuntu-latest`, `macos-latest`, `windows-latest`.
- [ ] Keep CI runtime bounded (smoke tests for wide matrix, fuller tests on Linux).
- [ ] Ensure artifact and cache keys include OS + Python version.

### 4) Versioning + Release Automation

- [ ] Adopt semantic versioning policy and changelog flow.
- [ ] Add tag-triggered release workflow.
- [ ] Ensure release notes are generated from commits/PRs.
- [ ] Add guardrails so only valid tags trigger publish.

### 5) Trusted PyPI Publishing

- [ ] Configure PyPI Trusted Publisher for this repository.
- [ ] Add publish workflow triggered on release tag.
- [ ] Build and upload sdist + wheel.
- [ ] Add post-publish sanity check (install from PyPI and import smoke test).

## Execution Order (Recommended)

Run these in order to reduce rework:

1. **Packaging extras + docs skeleton**
2. **CI matrix expansion (without publish)**
3. **Docker build definitions + CI build validation**
4. **Semantic release/tag workflow**
5. **Trusted Publisher wiring + dry run + first real tag**

## GitHub Issue Breakdown (Ready to Create)

Use these as issue titles:

1. `infra: add pip extras groups (rl/science/dev/all)`
2. `docs: document installation modes and extras`
3. `ci: add Python 3.9-3.12 compatibility matrix`
4. `ci: add Linux/macOS/Windows matrix smoke jobs`
5. `infra: add full training Docker image build`
6. `infra: add CPU-only Docker image build`
7. `ci: publish Docker images on release tag`
8. `release: adopt semver and tag-driven release workflow`
9. `release: configure Trusted Publisher for PyPI`
10. `release: publish wheel + sdist on tag with smoke test`

## Release Readiness Checklist (Gate)

All items below should be green before cutting vNext:

- [ ] `pip install exorl[rl]` succeeds in clean environment.
- [ ] `pip install exorl[science]` succeeds in clean environment.
- [ ] `pip install exorl[all]` succeeds in clean environment.
- [ ] Docker full image builds and runs example command.
- [ ] Docker CPU-only image builds and runs CI smoke command.
- [ ] CI matrix passes for all target Python and OS combinations.
- [ ] Release workflow creates changelog/release notes as expected.
- [ ] PyPI publish succeeds from tag using Trusted Publisher.
- [ ] Fresh environment can `pip install exorl` and import package.
- [ ] README install/deployment sections reflect final commands and tags.

## Nice-to-Have (Post vNext)

- [ ] Conda-forge recipe for scientific Python users.
- [ ] Nightly/pre-release channel for early adopters.
- [ ] SBOM/provenance attestations for release artifacts.

