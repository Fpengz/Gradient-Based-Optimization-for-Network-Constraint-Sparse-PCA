# Worktrees Guide

This repo uses **Git worktrees** to isolate paper phases and prevent branch cross-contamination.  
**Rule:** The **worktree folder name must exactly match the branch name**.

## Current Structure
- Main repo: `/Users/zhoufuwang/Projects/GRPCA-GD`
- Worktrees live under: `/Users/zhoufuwang/Projects/GRPCA-GD/.worktrees/`

## Naming Convention (Strict)
- Branch name: `paper-<track>-<phase>-<status>`
  - Example: `paper-trackA-phase2-final`
- Worktree folder name **must equal** the branch name:
  - `/Users/zhoufuwang/Projects/GRPCA-GD/.worktrees/paper-trackA-phase2-final`

## Create a New Worktree (Correct Way)
```bash
# Example branch name
BRANCH=paper-trackA-phase3-draft

# Create branch and worktree (folder name matches branch)
git -C /Users/zhoufuwang/Projects/GRPCA-GD worktree add \
  /Users/zhoufuwang/Projects/GRPCA-GD/.worktrees/${BRANCH} \
  -b ${BRANCH}
```

## Move/Rename a Worktree (if misnamed)
```bash
OLD=/Users/zhoufuwang/Projects/GRPCA-GD/.worktrees/old-name
NEW=/Users/zhoufuwang/Projects/GRPCA-GD/.worktrees/paper-trackA-phase2-final

git -C /Users/zhoufuwang/Projects/GRPCA-GD worktree move "$OLD" "$NEW"
```

## Remove an Unused Worktree
```bash
WT=/Users/zhoufuwang/Projects/GRPCA-GD/.worktrees/paper-trackA-phase1-revision
git -C /Users/zhoufuwang/Projects/GRPCA-GD worktree remove "$WT"
```

## Verify Worktrees
```bash
git -C /Users/zhoufuwang/Projects/GRPCA-GD worktree list
```

## Notes
- Only use the main repo (`/Users/zhoufuwang/Projects/GRPCA-GD`) for reference.  
- **Active Track A Phase 2 worktree:**  
  `/Users/zhoufuwang/Projects/GRPCA-GD/.worktrees/paper-trackA-phase2-final`
