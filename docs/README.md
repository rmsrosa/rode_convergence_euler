# How to docs

## Necessary files

1. Have a `docs/` folder with a `make.jl` file and a `scr/` folder and at least an `index.md` file inside `scr/`;
2. The `make.jl` file should have, at least, `using Documenter` and a `makedocs(sitename="name of site")`.
3. The `index.md` can have whatever.
4. The packages using in the documentation can be in a `Project.toml` either in the root directory of the repository or within the `docs/`.
5. If the documention is for an actual package and your project is inside `docs/`, you should dev your own package with `dev ../MyPackage`.
6. If you don't actually have a package to document and only want to type some notes, just keep everything inside `docs/`.

## Directly with Documenter

1. cd to root or the docs directory;
2. activate root or docs project;
3. make sure you have `Documenter.jl` and all the other relevant packages added to the project;
3. create/update the html docs with `julia> include("docs/make.jl")`;
4. open the indicated local web page in a browser (usually [http://localhost:8000])

## Via LiveServer

1. Have `LiveServer.jl` installed in the global environment;
2. Do `using LiveServer`;
3. Activate the root or docs project;
4. Do `julia> servedocs()`;
5. open the indicated local web page in a browser (usually [http://localhost:8000])
