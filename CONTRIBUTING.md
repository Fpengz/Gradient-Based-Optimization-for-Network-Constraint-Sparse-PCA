# Contributing to NC-SPCA

## Commit Message Guidelines

We follow the **Conventional Commits** specification. This leads to more readable history that is easier to reason about.

### Format

Each commit message consists of a **header**, a **body**, and a **footer**. The header has a special format that includes a **type**, a **scope**, and a **subject**:

```text
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

### Type

Must be one of the following:

- **feat**: A new feature
- **fix**: A bug fix
- **chore**: Changes to the build process or auxiliary tools and libraries such as documentation generation
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests

### Scope

The scope could be anything specifying the place of the commit change.
Examples: `optim`, `models`, `data`, `experiments`, `deps`.

### Subject

The subject contains a succinct description of the change:

- Use the imperative, present tense: "change" not "changed" nor "changes"
- Don't capitalize the first letter
- No dot (.) at the end

### Body

The body should include the motivation for the change and contrast this with previous behavior.

### Footer

The footer should contain any information about **Breaking Changes** and is also the place to reference JIRA issues or GitHub issues that this commit closes.

Example:
```text
feat(optim): add accelerated proximal gradient method

This implements the FISTA algorithm to improve convergence speed
for sparse problems.

Closes JIRA-42
```

## Setup

To help adhere to this standard, you can configure git to use the provided template:

```bash
git config commit.template .gitmessage
```
