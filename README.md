# WatChMaL_analysis
Example repository for usage of the WatChMaL repository

# Working With Submodules

See for reference: https://git-scm.com/book/en/v2/Git-Tools-Submodules

## Key commands

To clone the repo with submodules included:

```
$ git clone --recurse-submodules https://github.com/Whisky-Jack/WatChMaL_analysis.git
```

To update the submodules you can:
1. Enter the submodule directory and run fetch then merge. e.g.

```
$ cd WatChMaL
$ git fetch
$ git merge origin/master
```

2. You can get Git to fetch updates for the submodule for you using

```
$ git submodule update --remote WatChMaL
```
