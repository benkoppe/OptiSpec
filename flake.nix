{
  description = "OptiSpec development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      perSystem =
        { pkgs, ... }:
        {
          devShells.default = pkgs.mkShell {
            packages = [
              pkgs.python3
              pkgs.uv
              pkgs.ruff
            ];

            env = {
              UV_PYTHON = "${pkgs.python3}/bin/python";
              UV_PYTHON_DOWNLOADS = "never";
            };

            # PyPI wheels (numpy, jaxlib) expect an FHS system; libcuda comes from the host driver.
            # The python setup hook's PYTHONPATH would shadow the uv venv.
            shellHook = ''
              unset PYTHONPATH
              export LD_LIBRARY_PATH=${
                pkgs.lib.makeLibraryPath [
                  pkgs.stdenv.cc.cc.lib
                  pkgs.zlib
                ]
              }:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
            '';
          };
        };
    };
}
