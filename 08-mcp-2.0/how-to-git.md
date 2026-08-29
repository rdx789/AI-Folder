# how-to-git — pushing / syncing this repo

## The situation

This `homework/` folder is its own standalone Git repo, created with `git init`
here (not a clone). Consequences:

- It has **no remote**. `git commit` works; `git push` fails with
  *"no configured push destination"* / *"no upstream configured for branch 'master'"*.
- The published copy of this work lives **inside another repo**:
  `github.com/rdx789/AI-Folder`, under the subfolder `08-mcp-2.0/`.
- Path mismatch: files sit at the **root** here, but under `08-mcp-2.0/` there.
  So a plain `git push` could never line the two up even if a remote were added.

As of the last sync the two are identical — this repo's `HEAD` tree SHA equals
`AI-Folder/master:08-mcp-2.0/`.

## How to get a local change onto GitHub

Pick one approach and stick to it.

### Option A — work in an AI-Folder clone instead (simplest)

Treat this folder as scratch. Do real work in a clone of the monorepo:

```sh
git clone https://github.com/rdx789/AI-Folder.git
cd AI-Folder/08-mcp-2.0
# edit, then:
git add -A && git commit -m "..." && git push
```

### Option B — push this repo's history into the subfolder with git subtree

From a clone of AI-Folder, pull this repo in under the `08-mcp-2.0/` prefix:

```sh
# one-time: from inside the AI-Folder clone
git remote add hw "/Users/dron/Courses/Software-Classes/AI-Engineer/Lesson-08 - MCP/lesson-08-mcp-2.0/homework"
git subtree pull --prefix=08-mcp-2.0 hw master
git push
```

Repeat the `subtree pull` + `push` after each batch of local commits here.

### Option C — give this repo its own GitHub repo

Only if you want this work tracked separately from the monorepo (they will then
drift unless you update both):

```sh
gh repo create rdx789/lesson-08-mcp --private --source=. --remote=origin --push
# afterwards: git push
```

## Checking whether the two are still in sync

```sh
git remote add ai-folder https://github.com/rdx789/AI-Folder.git
git fetch ai-folder master
# equal output = in sync:
git rev-parse HEAD^{tree}
git ls-tree ai-folder/master 08-mcp-2.0    # compare the tree SHA in column 3
git remote remove ai-folder
```
