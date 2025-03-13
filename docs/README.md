# How to docs

## Preliminaries

1. Make sure you have [Julia](https://julialang.org)  installed, the lates version 1.11.x preferably, but it should work in other recent versions;
2. Change directory to the root of the project;
3. Activate the `docs/` environment with `]activate docs`;
4. Instantiate the local package `RODEConvergeEuler` in dev mode with `]dev ../rode_convergence_euler`, or with any other directory name that you used to clone the repository into.
5. Make sure you have all the packages in the `docs/` environment instantiated by typing `]instantiate`.

## Generate the documentation directly with Documenter

1. Change directory to the root of the project
2. Execute `docs/make.jl` with either `julia> include("docs/make.jl")` (from the julia REPL) or `julia docs/make
.jl` (from the shell);
3. Open the `index.html` web page generated in the `docs/build/` directory with e.g. `open docs/build/index.html` (from the shell) or directly from a browser or from your folder manager;

## Generate documentation via LiveServer

1. Change directory to the root of the project
2. Execute `docs/liveserver.jl` with either `julia> include("docs/liveserver.jl")` (from the julia REPL) or `julia docs/liveserver.jl` (from the shell);
3. Hit `ENTER` to the prompt, to not execute the documentation in DRAFT mode, so that all the julia example blocks are evaluated (answer with "y" if you just want to read the text, with no code evaluation);
4. open the indicated local web page in a browser (usually [http://localhost:8000])

## Execute specific scripts

Each example of strong convergence is in a script in the folder `docs/literate`. You can execute each of them as a julia script, preferably in an IDE, like VSCode, so you can better visualize the generated plots.

For example, in VSCode, with the [Julia VSCode extension](https://www.julia-vscode.org/docs/latest/) installed, load the desired script, change the environment to `docs`, and hit SHIFT+COMMAND+ENTER (Mac) or SHIFT-ALT-ENTER (Linux or Windows) to execute the file.

Or copy and paste the code to the Julia REPL.

## Necessary files for general documentation

1. Have a `docs/` folder with a `make.jl` file and a `scr/` folder and at least an `index.md` file inside `scr/`;
2. The `make.jl` file should have, at least, `using Documenter` and a `makedocs(sitename="name of site")`.
3. The `index.md` can have whatever.
4. The packages using in the documentation can be in a `Project.toml` either in the root directory of the repository or within the `docs/`.
5. If the documention is for an actual package and your project is inside `docs/`, you should dev your own package with `dev ../MyPackage`.
6. If you don't actually have a package to document and only want to type some notes, just keep everything inside `docs/`.
