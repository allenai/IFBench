# Publishing a new version of `ifbench` to PyPI

Releases are published to PyPI automatically via GitHub Actions
([.github/workflows/release.yml](.github/workflows/release.yml)) using
[PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC).
No API tokens are stored anywhere.

## One-time setup (already done — skip unless re-bootstrapping)

1. Create the project on PyPI by uploading the first release manually
   (`uv build && uv publish`), or have a PyPI admin reserve the name.
2. On PyPI → project → **Publishing** → **Add a pending publisher**:
   - Owner: `allenai`
   - Repository name: `IFBench`
   - Workflow filename: `release.yml`
   - Environment name: `pypi`
3. In GitHub → repo **Settings** → **Environments** → create an environment
   named `pypi`. Optionally require reviewers / restrict to tags `v*`.

## Cutting a release

1. Make sure `main` is green and contains everything you want shipped.
2. Bump the version in `pyproject.toml`:
   ```toml
   version = "0.3.0"
   ```
   Follow [semver](https://semver.org/): patch for bugfixes, minor for new
   verifiers / additive registry entries, major for breaking API changes
   (e.g., removing or renaming a registry key).
3. Run the full test suite locally:
   ```bash
   uv sync
   uv run python instructions_test.py
   ```
4. Build locally and sanity-check the wheel contents:
   ```bash
   uv build
   uv run python -c "import zipfile; z=zipfile.ZipFile('dist/ifbench-0.3.0-py3-none-any.whl'); print('\n'.join(z.namelist()))"
   ```
   Confirm `ifbench/data/IFBench_test.jsonl` is present.
5. Commit and push:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.3.0"
   git push
   ```
6. Tag and push the tag — this is what triggers the workflow:
   ```bash
   git tag v0.3.0
   git push origin v0.3.0
   ```
   The tag **must** start with `v` to match the workflow's `on.push.tags`
   filter, and **must** match the version in `pyproject.toml`.
7. Watch the run at
   `https://github.com/allenai/IFBench/actions/workflows/release.yml`.
   If the `pypi` environment requires approval, approve it.
8. Verify the release went live:
   ```bash
   uv run --with "ifbench==0.3.0" python -c "from ifbench import instructions_registry; print(len(instructions_registry.INSTRUCTION_DICT))"
   ```
   Expect `83` (or whatever the new count is).
9. Create a GitHub Release from the tag with release notes.

## If the publish fails

- **`File already exists`** on PyPI: you can't overwrite an existing version.
  Bump to the next patch, retag, push.
- **OIDC / trusted publisher error**: verify the PyPI publisher config
  matches the repo/workflow/environment names exactly, and that the workflow
  ran from a tag on the default branch.
- **Tests fail in CI**: fix on `main`, delete the bad tag locally and
  remotely (`git tag -d v0.3.0 && git push --delete origin v0.3.0`), then
  retag.

## Yanking a bad release

If a published version is broken, do **not** try to overwrite it. Either:

- Yank it on PyPI (project → Manage → Releases → Yank) — the version stays
  installable for existing pins but new resolutions skip it.
- Cut a fixed `0.3.1` and tell downstreams to upgrade.
