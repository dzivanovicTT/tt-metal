#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
tt-run - MPI process launcher for TT-Metal and TTNN distributed applications

tt-run is a lightweight MPI process launcher for TT-Metal and TTNN distributed applications,
providing automatic rank-to-fabric binding and environment setup. It simplifies launching
multi-host applications by mapping MPI ranks to mesh fabric coordinates (MeshId, HostRankId).

Features:
- Automatic Rank-to-Fabric Mapping: Maps MPI ranks to (MeshId, HostRankId) pairs
- Environment Setup: Configures per-rank environment variables including device visibility
- Flexible Configuration: Supports YAML format for rank bindings
- Multi-Mesh Support: Handles both single large mesh and inter-mesh configurations
- Multi-Application Support: Different ranks can run different executables
- Remote Deployment: Automatically deploy binaries and configs to remote hosts via SSH/SCP
- OpenMPI Support: Uses OpenMPI's multi-app context syntax for per-rank environment

Quick Start:
    # Launch with rank binding configuration (defaults to localhost)
    tt-run --rank-binding rank_binding.yaml ./my_app

    # Launch on multiple hosts with rankfile
    tt-run --rankfile hosts.txt --rank-binding binding.yaml ./my_app

    # Dry run to see generated command
    tt-run --rank-binding binding.yaml --dry-run ./my_app

    # Deploy binaries and run
    tt-run --rankfile hosts.txt --rank-binding binding_with_deployment.yaml ./my_app

    # Deploy only (useful for testing deployment)
    tt-run --rankfile hosts.txt --rank-binding binding.yaml --deploy-only

Configuration Example:
    # rank_binding.yaml
    rank_bindings:
      - rank: 0
        mesh_id: 0
        host_rank_id: 0
        env_overrides:
          TT_METAL_VISIBLE_DEVICES: "0,1,4,5"

      - rank: 1
        mesh_id: 0
        host_rank_id: 1
        env_overrides:
          TT_METAL_VISIBLE_DEVICES: "2,3,6,7"

      # Optional: Different program for specific ranks
      - rank: 2
        mesh_id: 0
        host_rank_id: 2
        program: ["aggregator", "-c", "config.yaml"]

    global_env:
      TT_LOGGER_LEVEL: Debug

    mesh_graph_path: "path/to/mesh_graph_descriptor.yaml"

Understanding Mesh Topology:
    The mesh_id and host_rank_id values map to the TT-Metal fabric topology as defined
    in mesh graph descriptor files. For example, in a T3K dual-host configuration:

    - mesh_id=0 identifies the mesh (useful for multi-mesh setups)
    - host_rank_id=0 controls chips 0,1,2,3 (first board)
    - host_rank_id=1 controls chips 4,5,6,7 (second board)

    The mesh graph descriptor defines the chip connectivity and host rank assignments.
    See tt_metal/fabric/mesh_graph_descriptors/ for examples.

Implementation Notes:
    This tool uses OpenMPI's multi-app context syntax to set different environment
    variables for each rank without requiring wrapper scripts. The generated command
    structure uses the -x flag to pass environment variables directly to each rank.
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
import socket

import click
from pydantic import BaseModel, Field, field_validator
from pydantic import ValidationError
import yaml


class FileDeployment(BaseModel):
    """File deployment specification.

    Defines which files to deploy to remote hosts and where to place them.

    Attributes:
        source: Local file or directory path to deploy
        destination: Remote destination path (absolute or relative to remote working directory)
        preserve_path: If True, preserves directory structure for directory sources
    """

    source: str = Field(..., description="Local file or directory path")
    destination: str = Field(..., description="Remote destination path")
    preserve_path: bool = Field(True, description="Preserve directory structure for directories")


class DeploymentConfig(BaseModel):
    """Remote deployment configuration.

    Defines how to deploy files to remote hosts before launching MPI jobs.

    Attributes:
        enabled: Whether deployment is enabled
        ssh_user: SSH username for remote hosts (defaults to current user)
        files: List of files/directories to deploy
        create_dirs: Whether to create destination directories if they don't exist
        skip_local: Whether to skip deployment to localhost
    """

    enabled: bool = Field(True, description="Enable remote deployment")
    ssh_user: Optional[str] = Field(None, description="SSH username (defaults to current user)")
    files: List[FileDeployment] = Field(default_factory=list, description="Files to deploy")
    create_dirs: bool = Field(True, description="Create destination directories if needed")
    skip_local: bool = Field(True, description="Skip deployment to localhost")


class RankBinding(BaseModel):
    """Binding between MPI rank and fabric coordinates.

    Each MPI rank is mapped to a specific mesh and host rank within that mesh.
    This allows TT-Metal to understand the fabric topology and coordinate
    distributed operations across multiple hosts.

    Attributes:
        rank: MPI rank number (must be >= 0 and unique)
        mesh_id: Identifies which mesh this rank belongs to (for multi-mesh setups)
        host_rank_id: Identifies the host rank within the mesh
        env_overrides: Per-rank environment variable overrides (e.g., device visibility)
        program: Optional per-rank program override (path and arguments)
    """

    rank: int = Field(..., ge=0, description="MPI rank (must be >= 0)")
    mesh_id: int = Field(..., ge=0, description="Mesh ID")
    host_rank_id: int = Field(..., ge=0, description="Host rank ID within the mesh")
    env_overrides: Dict[str, str] = Field(default_factory=dict, description="Environment variable overrides")
    program: Optional[List[str]] = Field(None, description="Per-rank program override [executable, arg1, arg2, ...]")


class TTRunConfig(BaseModel):
    """Configuration for tt-run.

    This configuration defines how MPI ranks map to the TT-Metal fabric topology
    and what environment variables should be set for the distributed execution.

    Attributes:
        rank_bindings: List of rank-to-fabric mappings (must have at least one)
        global_env: Environment variables applied to all ranks
        mesh_graph_path: Path to mesh graph descriptor YAML file
        deployment: Optional remote deployment configuration

    Example YAML format:
        rank_bindings:
          - rank: 0
            mesh_id: 0
            host_rank_id: 0
            env_overrides:
              TT_METAL_VISIBLE_DEVICES: "0,1,4,5"
          - rank: 1
            mesh_id: 0
            host_rank_id: 1
            env_overrides:
              TT_METAL_VISIBLE_DEVICES: "2,3,6,7"
          - rank: 2
            mesh_id: 0
            host_rank_id: 2
            program: ["nano_gpt_aggregator", "-c", "config.yaml"]
          - rank: 3
            mesh_id: 0
            host_rank_id: 3
            program: ["nano_gpt_optimizer", "-c", "config.yaml"]
        global_env:
          TT_LOGGER_LEVEL: Debug
          PYTHONPATH: "${PWD}"
        mesh_graph_path: "path/to/mesh_graph.yaml"

        # Optional deployment configuration
        deployment:
          enabled: true
          ssh_user: ttuser
          files:
            - source: "./build/nano_gpt"
              destination: "/home/ttuser/tt-metal/build/nano_gpt"
            - source: "./configs/training.yaml"
              destination: "/home/ttuser/tt-metal/configs/training.yaml"
    """

    rank_bindings: List[RankBinding] = Field(..., min_length=1, description="Rank to fabric bindings")
    global_env: Dict[str, str] = Field(default_factory=dict, description="Global environment variables")
    mesh_graph_path: Optional[str] = Field(None, description="Path to mesh graph descriptor")
    deployment: Optional[DeploymentConfig] = Field(None, description="Remote deployment configuration")

    @field_validator("rank_bindings")
    def validate_unique_ranks(cls, bindings):
        """Ensure all ranks are unique"""
        ranks = [b.rank for b in bindings]
        if len(ranks) != len(set(ranks)):
            raise ValueError("Duplicate ranks found in bindings")
        return bindings

    @field_validator("rank_bindings")
    def validate_contiguous_ranks(cls, bindings):
        """Ensure ranks are contiguous starting from 0"""
        ranks = sorted([b.rank for b in bindings])
        expected = list(range(len(ranks)))
        if ranks != expected:
            raise ValueError(f"Ranks must be contiguous from 0. Got: {ranks}")
        return bindings


def get_unique_hosts_from_rankfile(rankfile_path: Path) -> Set[str]:
    """Extract unique hostnames from MPI rankfile.

    Parses OpenMPI rankfile format to get list of unique hosts.
    Supports both simple format (hostname slots=N) and rank-specific format.

    Args:
        rankfile_path: Path to MPI rankfile

    Returns:
        Set of unique hostnames
    """
    hosts = set()
    with open(rankfile_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Handle "hostname slots=N" format
            if "slots=" in line:
                hostname = line.split()[0]
                hosts.add(hostname)
            # Handle "rank N=hostname slot=N" format
            elif "=" in line and "rank" in line:
                parts = line.split("=")
                if len(parts) >= 2:
                    hostname = parts[1].split()[0]
                    hosts.add(hostname)
    return hosts


def get_current_hostname() -> str:
    """Get the current machine's hostname."""
    return socket.gethostname()


def is_local_host(hostname: str) -> bool:
    """Check if a hostname refers to the local machine.

    Args:
        hostname: Hostname to check

    Returns:
        True if hostname is localhost, 127.0.0.1, or current hostname
    """
    local_names = {"localhost", "127.0.0.1", get_current_hostname()}
    try:
        # Also check if hostname resolves to localhost
        addr = socket.gethostbyname(hostname)
        if addr == "127.0.0.1":
            local_names.add(hostname)
    except socket.error:
        pass
    return hostname in local_names


def deploy_file_ssh(source: Path, destination: str, hostname: str, ssh_user: Optional[str] = None) -> None:
    """Deploy a file or directory to a remote host via SSH.

    Args:
        source: Local source path
        destination: Remote destination path
        hostname: Remote hostname
        ssh_user: SSH username (defaults to current user)

    Raises:
        subprocess.CalledProcessError: If SSH/SCP command fails
    """
    user_prefix = f"{ssh_user}@" if ssh_user else ""
    remote_spec = f"{user_prefix}{hostname}:{destination}"

    # Use scp with recursive flag for directories
    scp_cmd = ["scp", "-r", "-p", str(source), remote_spec]

    result = subprocess.run(scp_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, scp_cmd, output=result.stdout, stderr=result.stderr)


def create_remote_directory(directory: str, hostname: str, ssh_user: Optional[str] = None) -> None:
    """Create a directory on a remote host via SSH.

    Args:
        directory: Directory path to create
        hostname: Remote hostname
        ssh_user: SSH username (defaults to current user)

    Raises:
        subprocess.CalledProcessError: If SSH command fails
    """
    user_prefix = f"{ssh_user}@" if ssh_user else ""
    ssh_cmd = ["ssh", f"{user_prefix}{hostname}", f"mkdir -p '{directory}'"]

    result = subprocess.run(ssh_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, ssh_cmd, output=result.stdout, stderr=result.stderr)


def deploy_files_to_hosts(config: TTRunConfig, hosts: Set[str], verbose: bool = False) -> None:
    """Deploy files to remote hosts based on deployment configuration.

    Args:
        config: TTRun configuration with deployment settings
        hosts: Set of hostnames to deploy to
        verbose: Whether to print verbose output

    Raises:
        click.ClickException: If deployment fails
    """
    if not config.deployment or not config.deployment.enabled:
        return

    if not config.deployment.files:
        return

    ssh_user = config.deployment.ssh_user

    # Filter out local hosts if skip_local is enabled
    if config.deployment.skip_local:
        remote_hosts = {h for h in hosts if not is_local_host(h)}
    else:
        remote_hosts = hosts

    if not remote_hosts:
        if verbose:
            click.echo("[tt-run] No remote hosts to deploy to")
        return

    if verbose:
        click.echo(f"[tt-run] Deploying files to {len(remote_hosts)} remote hosts...")

    # Deploy to each remote host
    for hostname in remote_hosts:
        if verbose:
            click.echo(f"[tt-run] Deploying to {hostname}...")

        try:
            # Create destination directories if needed
            if config.deployment.create_dirs:
                dest_dirs = set()
                for file_spec in config.deployment.files:
                    dest_path = Path(file_spec.destination)
                    # If destination is a file, get its parent directory
                    if dest_path.suffix:  # Has file extension
                        dest_dirs.add(str(dest_path.parent))
                    else:
                        dest_dirs.add(str(dest_path))

                for dest_dir in dest_dirs:
                    create_remote_directory(dest_dir, hostname, ssh_user)

            # Deploy each file
            for file_spec in config.deployment.files:
                source_path = Path(file_spec.source).expanduser()
                if not source_path.exists():
                    raise click.ClickException(f"Source file not found: {source_path}")

                if verbose:
                    click.echo(f"  {source_path} -> {file_spec.destination}")

                deploy_file_ssh(source_path, file_spec.destination, hostname, ssh_user)

            if verbose:
                click.echo(f"[tt-run] ✓ Deployment to {hostname} complete")

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to deploy to {hostname}: {e.stderr}"
            if verbose:
                click.echo(f"[tt-run] ✗ {error_msg}", err=True)
            raise click.ClickException(error_msg)
        except Exception as e:
            error_msg = f"Failed to deploy to {hostname}: {str(e)}"
            if verbose:
                click.echo(f"[tt-run] ✗ {error_msg}", err=True)
            raise click.ClickException(error_msg)

    if verbose:
        click.echo("[tt-run] ✓ All deployments complete")


def parse_binding_config(yaml_path: str) -> TTRunConfig:
    """Parse YAML configuration file with schema validation.

    Expected YAML format:
        rank_bindings:
          - rank: 0
            mesh_id: 0
            host_rank_id: 0
            env_overrides:
              KEY: value
          - rank: 1
            mesh_id: 0
            host_rank_id: 1
            env_overrides:
              KEY: value
        global_env:
          KEY: value
        mesh_graph_path: path/to/mesh.yaml

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        TTRunConfig: Validated configuration object

    Raises:
        ValueError: If configuration is invalid or file cannot be parsed
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    try:
        return TTRunConfig(**data)
    except ValidationError as e:
        raise ValueError(f"Invalid configuration: {e}")


def build_mpi_command(
    config: TTRunConfig, program: List[str], rankfile: Optional[Path] = None, mpi_args: Optional[List[str]] = None
) -> List[str]:
    """Build OpenMPI command with per-rank environment variables and programs.

    Uses OpenMPI's multi-app context syntax to set different environment
    variables and optionally different programs for each rank. This approach
    avoids the need for wrapper scripts by using the -x flag to pass
    environment variables directly.

    The generated command structure is:
        mpirun [rankfile] [mpi_args] \
            -np 1 -x ENV1=val1 -x ENV2=val2 program1 [args] : \
            -np 1 -x ENV1=val3 -x ENV2=val4 program2 [args]

    Environment variables set for each rank:
        - TT_METAL_MESH_ID: The mesh this rank belongs to
        - TT_METAL_HOST_RANK_ID: The host rank within the mesh
        - TT_METAL_CACHE: Per-rank cache directory
        - TT_METAL_HOME: TT-Metal installation directory
        - PYTHONPATH: Python module search path
        - TT_METAL_MESH_GRAPH_PATH: Path to mesh graph descriptor (if provided)
        - Plus any global_env and rank-specific env_overrides

    Each rank can optionally specify its own program via the 'program' field
    in the rank binding. If not specified, the global program is used.

    Args:
        config: Validated configuration with rank bindings
        program: Default program command and arguments to execute
        rankfile: Optional MPI rankfile for host placement
        mpi_args: Additional arguments to pass to mpirun

    Returns:
        List of command arguments for subprocess.run()
    """
    cmd = ["mpirun"]

    # Add rankfile if provided
    if rankfile:
        cmd.extend(["--rankfile", str(rankfile)])

    # Add any additional MPI arguments first
    if mpi_args:
        cmd.extend(mpi_args)

    # Get standard environment variables
    standard_env = {
        "PYTHONPATH": os.environ.get("PYTHONPATH", os.getcwd()),
        "TT_METAL_HOME": os.environ.get("TT_METAL_HOME", os.getcwd()),
    }

    # Build per-rank application contexts
    for i, binding in enumerate(sorted(config.rank_bindings, key=lambda b: b.rank)):
        if i > 0:
            cmd.append(":")

        cmd.extend(["-np", "1"])

        # Add standard environment variables
        for key, value in standard_env.items():
            cmd.extend(["-x", f"{key}={value}"])

        # Add per-rank cache directory
        cache_dir = os.environ.get(
            "TT_METAL_CACHE", f"{os.path.expanduser('~')}/.cache/{os.uname().nodename}_rank{binding.rank}"
        )
        cmd.extend(["-x", f"TT_METAL_CACHE={cache_dir}"])

        # Add global environment variables
        for key, value in config.global_env.items():
            expanded_value = os.path.expandvars(value)
            cmd.extend(["-x", f"{key}={expanded_value}"])

        # Add fabric configuration
        cmd.extend(["-x", f"TT_METAL_MESH_ID={binding.mesh_id}"])
        cmd.extend(["-x", f"TT_METAL_HOST_RANK_ID={binding.host_rank_id}"])

        if config.mesh_graph_path:
            cmd.extend(["-x", f"TT_METAL_MESH_GRAPH_PATH={config.mesh_graph_path}"])

        # Add rank-specific environment overrides
        for key, value in binding.env_overrides.items():
            expanded_value = os.path.expandvars(value)
            cmd.extend(["-x", f"{key}={expanded_value}"])

        # Add the program and its arguments
        # Use per-rank program if specified, otherwise use the global program
        if binding.program:
            cmd.extend(binding.program)
        else:
            cmd.extend(program)

    return cmd


def validate_mpi_args(ctx, param, value):
    """Click callback to parse MPI arguments"""
    if value:
        return shlex.split(value)
    return None


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option("--rankfile", type=click.Path(exists=True, path_type=Path), help="OpenMPI rankfile for host placement")
@click.option(
    "--rank-binding",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Rank binding configuration file (YAML)",
)
@click.option("--mesh-graph", type=click.Path(exists=True, path_type=Path), help="Mesh graph descriptor file")
@click.option("--dry-run", is_flag=True, help="Print command without executing")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.option("--mpi-args", callback=validate_mpi_args, help="Additional MPI arguments (quoted)")
@click.option("--deploy/--no-deploy", default=None, help="Enable/disable remote deployment (overrides config)")
@click.option("--deploy-only", is_flag=True, help="Only deploy files, don't run MPI job")
@click.pass_context
def main(ctx, rankfile, rank_binding, mesh_graph, dry_run, verbose, mpi_args, deploy, deploy_only):
    """tt-run - MPI process launcher for TT-Metal and TTNN distributed applications

    tt-run simplifies launching distributed TT-Metal applications by automatically
    mapping MPI ranks to mesh fabric coordinates and setting up the environment.

    Common Use Cases:

    Big Mesh (2x4 mesh across two hosts):
        tt-run --rankfile hosts.txt --rank-binding big_mesh.yaml ./my_app

    Inter-Mesh (two separate 2x2 meshes):
        tt-run --rankfile hosts.txt --rank-binding inter_mesh.yaml ./my_app

    Single Node Testing (simulating multi-host on one machine):
        tt-run --rank-binding single_node.yaml ./my_app

    Examples:

        # Single host, multiple processes
        tt-run --rank-binding rank_binding.yaml ./my_app

        # Multi-host with rankfile
        tt-run --rankfile hosts.txt --rank-binding binding.yaml ./my_app

        # With additional MPI args
        tt-run --rank-binding binding.yaml --mpi-args "--bind-to core" ./my_app

        # Dry run to see command
        tt-run --rank-binding binding.yaml --dry-run ./my_app

        # With remote deployment
        tt-run --rankfile hosts.txt --rank-binding deploy_config.yaml ./my_app

        # Deploy only mode
        tt-run --rankfile hosts.txt --rank-binding deploy_config.yaml --deploy-only

        # Override deployment setting
        tt-run --rank-binding config.yaml --no-deploy ./my_app

    Environment Variables:
        The following variables are automatically set for each rank:
        - TT_METAL_MESH_ID: Mesh identifier
        - TT_METAL_HOST_RANK_ID: Host rank within the mesh
        - TT_METAL_CACHE: Per-rank cache directory
        - TT_METAL_HOME: TT-Metal installation directory
        - PYTHONPATH: Python module search path

    See examples/ttrun/ for example configuration files.
    """
    # Get the program and its arguments from extra args
    program = ctx.args

    # Parse configuration
    try:
        config = parse_binding_config(str(rank_binding))
    except (ValueError, ValidationError) as e:
        raise click.ClickException(f"Configuration error: {e}")

    # Validate program specification
    # Either a global program must be provided, or all ranks must have their own programs
    ranks_with_programs = sum(1 for b in config.rank_bindings if b.program is not None)
    if not program and ranks_with_programs < len(config.rank_bindings):
        raise click.ClickException(
            "No program specified. Either provide a program as argument or specify programs for all ranks in the binding configuration."
        )

    # Override mesh graph if provided via command line
    if mesh_graph:
        config.mesh_graph_path = str(mesh_graph)

    # Handle deployment command-line overrides
    deployment_enabled = False
    if config.deployment:
        deployment_enabled = config.deployment.enabled

    # Command-line --deploy/--no-deploy overrides config
    if deploy is not None:
        deployment_enabled = deploy

    # Deploy files to remote hosts if enabled
    if deployment_enabled or deploy_only:
        # Get list of hosts from rankfile or use localhost
        if rankfile:
            hosts = get_unique_hosts_from_rankfile(rankfile)
        else:
            # Default to localhost if no rankfile
            hosts = {"localhost"}

        # Create temporary deployment config if needed
        if not config.deployment:
            click.echo("[tt-run] Warning: --deploy specified but no deployment configuration found", err=True)
        else:
            deploy_files_to_hosts(config, hosts, verbose)

    # Exit early if deploy-only mode
    if deploy_only:
        if verbose:
            click.echo("[tt-run] Deploy-only mode: skipping MPI launch")
        return

    # Build MPI command
    mpi_cmd = build_mpi_command(config, program, rankfile, mpi_args)

    if verbose or dry_run:
        # Pretty print the command for readability
        # This helps users understand the generated command structure,
        # especially with the multi-app context syntax
        if len(mpi_cmd) > 10:  # Long command
            click.echo("[tt-run] Command:")
            formatted_cmd = "mpirun \\"
            i = 1
            while i < len(mpi_cmd):
                if mpi_cmd[i] == ":":
                    formatted_cmd += " \\\n    : \\"
                else:
                    formatted_cmd += f"\n    {mpi_cmd[i]}"
                    if i + 1 < len(mpi_cmd) and mpi_cmd[i + 1] != ":":
                        formatted_cmd += " \\"
                i += 1
            click.echo(formatted_cmd)
        else:
            click.echo("[tt-run] Command: " + " ".join(mpi_cmd))

    if dry_run:
        return

    # Launch MPI job
    # Note: We use subprocess.run() without shell=True for security
    # The command list is already properly formatted for direct execution
    try:
        result = subprocess.run(mpi_cmd)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully with proper exit code (128 + SIGINT)
        click.echo("\n[tt-run] Interrupted", err=True)
        sys.exit(130)
    except OSError as e:
        raise click.ClickException(f"Error launching mpirun: {e}")


if __name__ == "__main__":
    main()
