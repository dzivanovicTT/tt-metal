# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import tempfile
import os
import yaml
from pathlib import Path
from ttnn.distributed.ttrun import (
    RankBinding,
    TTRunConfig,
    parse_binding_config,
    build_mpi_command,
)


class TestBindingParser:
    """Test rank binding configuration parsing"""

    def test_parse_yaml_explicit_format(self):
        """Test parsing YAML with explicit bindings format"""
        yaml_content = """
bindings:
  - rank: 0
    mesh_id: 0
    host_rank_id: 0
    env_overrides:
      TEST_VAR: "value0"
  - rank: 1
    mesh_id: 0
    host_rank_id: 1
    env_overrides:
      TEST_VAR: "value1"
global_env:
  GLOBAL_VAR: "global_value"
mesh_graph_path: "test_mesh.yaml"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = parse_binding_config(f.name)

            assert len(config.bindings) == 2
            assert config.bindings[0].rank == 0
            assert config.bindings[0].mesh_id == 0
            assert config.bindings[0].host_rank_id == 0
            assert config.bindings[0].env_overrides == {"TEST_VAR": "value0"}
            assert config.bindings[1].rank == 1
            assert config.bindings[1].mesh_id == 0
            assert config.bindings[1].host_rank_id == 1
            assert config.global_env == {"GLOBAL_VAR": "global_value"}
            assert config.mesh_graph_path == "test_mesh.yaml"

            os.unlink(f.name)

    def test_validation_errors(self):
        """Test validation errors for invalid configurations"""
        # Test duplicate ranks
        yaml_content = """
bindings:
  - rank: 0
    mesh_id: 0
    host_rank_id: 0
  - rank: 0
    mesh_id: 0
    host_rank_id: 1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            with pytest.raises(ValueError, match="Duplicate ranks"):
                parse_binding_config(f.name)

            os.unlink(f.name)

        # Test non-contiguous ranks
        yaml_content = """
bindings:
  - rank: 0
    mesh_id: 0
    host_rank_id: 0
  - rank: 2
    mesh_id: 0
    host_rank_id: 1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            with pytest.raises(ValueError, match="Ranks must be contiguous"):
                parse_binding_config(f.name)

            os.unlink(f.name)


class TestMPICommand:
    """Test MPI command building"""

    def test_build_mpi_command_basic(self):
        """Test basic MPI command building"""
        config = TTRunConfig(
            bindings=[
                RankBinding(rank=0, mesh_id=0, host_rank_id=0),
                RankBinding(rank=1, mesh_id=0, host_rank_id=1),
            ]
        )
        program = ["python", "test.py", "--arg"]

        cmd = build_mpi_command(config, program)

        assert cmd[0] == "mpirun"
        # Check for multi-app context with colon separator
        assert ":" in cmd
        # Check that each rank gets -np 1
        assert cmd.count("-np") == 2
        assert cmd.count("1") >= 2  # At least two "1"s for -np 1
        # Check environment variables
        assert "-x" in cmd
        assert any("TT_METAL_MESH_ID=0" in arg for arg in cmd)
        assert any("TT_METAL_HOST_RANK_ID=0" in arg for arg in cmd)
        assert any("TT_METAL_HOST_RANK_ID=1" in arg for arg in cmd)

    def test_build_mpi_command_with_rankfile(self):
        """Test MPI command building with rankfile"""
        config = TTRunConfig(
            bindings=[
                RankBinding(rank=0, mesh_id=0, host_rank_id=0),
            ]
        )
        program = ["./my_app"]
        rankfile = Path("/tmp/rankfile.txt")

        cmd = build_mpi_command(config, program, rankfile=rankfile)

        assert "--rankfile" in cmd
        assert str(rankfile) in cmd

    def test_build_mpi_command_with_env_overrides(self):
        """Test MPI command with environment overrides"""
        config = TTRunConfig(
            bindings=[
                RankBinding(rank=0, mesh_id=0, host_rank_id=0, env_overrides={"TEST_VAR": "value0", "DEVICE": "0,1"}),
            ],
            global_env={"GLOBAL": "test"},
            mesh_graph_path="/path/to/mesh.yaml",
        )
        program = ["./test"]

        cmd = build_mpi_command(config, program)

        # Check global env
        assert any("GLOBAL=test" in arg for arg in cmd)
        # Check rank-specific env
        assert any("TEST_VAR=value0" in arg for arg in cmd)
        assert any("DEVICE=0,1" in arg for arg in cmd)
        # Check mesh graph path
        assert any("TT_METAL_MESH_GRAPH_PATH=/path/to/mesh.yaml" in arg for arg in cmd)
