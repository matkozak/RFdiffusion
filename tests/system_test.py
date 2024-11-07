import unittest
import torch
import dgl
import logging
import os
import numpy as np
from typing import Optional
from se3_transformer.model import SE3Transformer
from se3_transformer.model.fiber import Fiber


class BaseTestCase(unittest.TestCase):
    """Base test class providing common setup and utilities."""

    @classmethod
    def setUpClass(cls):
        """Initialize common attributes and logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        cls.logger = logging.getLogger(cls.__name__)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setUp(self):
        """Set up test environment before each test."""
        torch.manual_seed(42)  # Reproducibility
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def tearDown(self):
        """Clean up after each test."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SystemCompatibilityTests(BaseTestCase):
    """Tests for system-level compatibility and environment setup."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._log_system_info()

    @classmethod
    def _log_system_info(cls):
        """Log system information for diagnostic purposes."""
        cls.logger.info("System Information:")
        info = {
            "PyTorch": torch.__version__,
            "DGL": dgl.__version__,
            "CUDA Available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            cuda_info = {
                "CUDA version": torch.version.cuda,
                "GPU Device": torch.cuda.get_device_name(),
                "cuDNN Version": torch.backends.cudnn.version(),
            }
            info.update(cuda_info)

            cuda_vars = ["CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH"]
            env_info = {var: os.environ.get(var, "Not set") for var in cuda_vars}
            info.update(env_info)

        for key, value in info.items():
            cls.logger.info(f"{key}: {value}")

    def test_cuda_availability(self):
        """Verify CUDA operations and memory management."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        initial_memory = torch.cuda.memory_allocated()

        # Test basic operations
        x = torch.randn(1000, 1000, device=self.device)
        y = torch.matmul(x, x.t())

        self.assertTrue(
            torch.isfinite(y).all(), "CUDA computation produced invalid values"
        )
        self.assertTrue(y.is_cuda, "Tensor not on CUDA")

        mid_memory = torch.cuda.memory_allocated()
        self.assertGreater(
            mid_memory, initial_memory, "Memory not allocated as expected"
        )

        # Test memory cleanup
        del x, y
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        self.assertLess(final_memory, mid_memory, "Memory not properly freed")


class LibraryCompatibilityTests(BaseTestCase):
    """Tests for cross-library compatibility between PyTorch, DGL, and NumPy."""

    def setUp(self):
        super().setUp()
        self.test_graph = self._create_test_graph()

    def _create_test_graph(self) -> dgl.DGLGraph:
        """Create a simple test graph with features."""
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 0])
        g = dgl.graph((src, dst))
        g.ndata["feat"] = torch.randn(3, 4)
        return g

    def test_dgl_cuda_compatibility(self):
        """Test DGL-CUDA interaction and message passing."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        g = self.test_graph.to(self.device)
        self.assertTrue(
            g.device.type == "cuda", f"Graph not on CUDA. Device: {g.device}"
        )

        # Test message passing
        with g.local_scope():
            g.ndata["h"] = torch.ones(g.num_nodes(), 5, device=self.device)
            g.update_all(
                message_func=dgl.function.copy_u("h", "m"),
                reduce_func=dgl.function.sum("m", "h_sum"),
            )
            out = g.ndata["h_sum"]
            self.assertTrue(out.is_cuda, "Message passing output not on CUDA")
            self.assertTrue(torch.all(out > 0), "Invalid message passing results")

    def test_numpy_interoperability(self):
        """Test NumPy interoperability with PyTorch and DGL."""
        # Test NumPy <-> PyTorch conversion
        np_array = np.random.randn(10, 10).astype(np.float32)
        torch_tensor = torch.from_numpy(np_array)
        self.assertTrue(
            torch.allclose(torch_tensor, torch.tensor(np_array)),
            "NumPy to PyTorch conversion failed",
        )

        # Test roundtrip conversion
        np_roundtrip = torch_tensor.numpy()
        self.assertTrue(
            np.allclose(np_array, np_roundtrip), "PyTorch to NumPy conversion failed"
        )

        # Test DGL with NumPy features
        g = self.test_graph
        np_features = np.random.randn(3, 4).astype(np.float32)
        g.ndata["numpy_feat"] = torch.from_numpy(np_features)
        retrieved_features = g.ndata["numpy_feat"].numpy()
        self.assertTrue(
            np.allclose(np_features, retrieved_features),
            "DGL NumPy feature roundtrip failed",
        )


class ModelCompatibilityTests(BaseTestCase):
    """Tests for SE3Transformer model compatibility and equivariance properties."""

    # Constants for equivariance testing
    EQUIV_TOL = 1e-3
    NUM_NODES = 512
    NUM_CHANNELS = 32

    @staticmethod
    def _create_rotation_matrix(angles: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Create a 3D rotation matrix using ZYZ Euler angles."""
        if angles is None:
            angles = torch.rand(3)

        def rot_z(gamma):
            return torch.tensor(
                [
                    [torch.cos(gamma), -torch.sin(gamma), 0],
                    [torch.sin(gamma), torch.cos(gamma), 0],
                    [0, 0, 1],
                ],
                dtype=gamma.dtype,
            )

        def rot_y(beta):
            return torch.tensor(
                [
                    [torch.cos(beta), 0, torch.sin(beta)],
                    [0, 1, 0],
                    [-torch.sin(beta), 0, torch.cos(beta)],
                ],
                dtype=beta.dtype,
            )

        alpha, beta, gamma = angles
        return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)

    def _create_random_graph(self, num_nodes: int) -> dgl.DGLGraph:
        """Create a random graph with given number of nodes."""
        edges_per_node = 18
        g = dgl.remove_self_loop(dgl.rand_graph(num_nodes, num_nodes * edges_per_node))
        return g

    def _assign_relative_positions(
        self, graph: dgl.DGLGraph, coords: torch.Tensor
    ) -> dgl.DGLGraph:
        """Assign relative position vectors to graph edges."""
        src, dst = graph.edges()
        graph.edata["rel_pos"] = coords[src] - coords[dst]
        return graph

    def _get_test_model(
        self, pooling: Optional[str] = None, return_type: Optional[int] = None
    ) -> SE3Transformer:
        """Create a test SE3Transformer model."""
        return SE3Transformer(
            num_layers=4,
            fiber_in=Fiber.create(2, self.NUM_CHANNELS),
            fiber_hidden=Fiber.create(3, self.NUM_CHANNELS),
            fiber_out=Fiber.create(2, self.NUM_CHANNELS),
            fiber_edge=Fiber({}),
            num_heads=8,
            channels_div=2,
            pooling=pooling,
            return_type=return_type,
        )

    def _get_model_outputs(
        self, model: SE3Transformer, rotation: torch.Tensor
    ) -> tuple[dict, dict]:
        """Get model outputs for original and rotated inputs."""
        # Create input features
        feats0 = torch.randn(self.NUM_NODES, self.NUM_CHANNELS, 1)
        feats1 = torch.randn(self.NUM_NODES, self.NUM_CHANNELS, 3)
        coords = torch.randn(self.NUM_NODES, 3)
        graph = self._create_random_graph(self.NUM_NODES)

        # Get outputs for original and rotated inputs
        graph1 = self._assign_relative_positions(graph, coords)
        out1 = model(graph1, {"0": feats0, "1": feats1}, {})

        graph2 = self._assign_relative_positions(graph, coords @ rotation)
        out2 = model(graph2, {"0": feats0, "1": feats1 @ rotation}, {})

        return out1, out2

    def test_se3_transformer_import(self):
        """Verify SE3Transformer can be imported and initialized."""
        self.assertIsNotNone(SE3Transformer, "Failed to import SE3Transformer")

    def test_basic_equivariance(self):
        """Test SE3Transformer's equivariance properties under rotation."""
        model = self._get_test_model()
        rotation = self._create_rotation_matrix()
        out1, out2 = self._get_model_outputs(model, rotation)

        # Type-0 features should be invariant under rotation
        self.assertTrue(
            torch.allclose(out2["0"], out1["0"], atol=self.EQUIV_TOL),
            "Type-0 features are not rotation invariant",
        )

        # Type-1 features should be equivariant under rotation
        self.assertTrue(
            torch.allclose(out2["1"], (out1["1"] @ rotation), atol=self.EQUIV_TOL),
            "Type-1 features are not rotation equivariant",
        )

    def test_pooled_equivariance(self):
        """Test equivariance of pooled type-1 features."""
        model = self._get_test_model(pooling="avg", return_type=1)
        rotation = self._create_rotation_matrix()
        out1, out2 = self._get_model_outputs(model, rotation)

        self.assertTrue(
            torch.allclose(out2, (out1 @ rotation), atol=self.EQUIV_TOL),
            "Pooled type-1 features are not rotation equivariant",
        )

    def test_pooled_invariance(self):
        """Test invariance of pooled type-0 features."""
        model = self._get_test_model(pooling="avg", return_type=0)
        rotation = self._create_rotation_matrix()
        out1, out2 = self._get_model_outputs(model, rotation)

        self.assertTrue(
            torch.allclose(out2, out1, atol=self.EQUIV_TOL),
            "Pooled type-0 features are not rotation invariant",
        )


def run_tests(verbosity: Optional[int] = 2):
    """Run all test suites with specified verbosity."""
    test_loader = unittest.TestLoader()
    test_suites = [
        test_loader.loadTestsFromTestCase(SystemCompatibilityTests),
        test_loader.loadTestsFromTestCase(LibraryCompatibilityTests),
        test_loader.loadTestsFromTestCase(ModelCompatibilityTests),
    ]

    combined_suite = unittest.TestSuite(test_suites)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(combined_suite)


if __name__ == "__main__":
    run_tests()
