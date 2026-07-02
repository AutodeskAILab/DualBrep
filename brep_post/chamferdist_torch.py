import torch

CHECK_CORRECTNESS = False
if CHECK_CORRECTNESS:
    try:
        from chamferdist import ChamferDistance as ChamferDistanceLib
    except ImportError:
        raise ImportError("Please install the chamferdist package to check correctness.")


class ChamferDistanceTorch:
    """
    A class to compute Chamfer Distance between two point clouds, supporting batch processing,
    one-way and two-way modes.
    """

    def __init__(self, use_squared_cdist=True):
        """
        Initialize ChamferDistance class.

        Args:
            use_squared_cdist: bool - If True, calculate squared Euclidean distance. If False, use Euclidean distance.
        """
        self.use_squared_cdist = use_squared_cdist

    def squared_cdist(self, pc1, pc2):
        """
        Compute the squared Euclidean distance matrix between two point clouds for a batch.

        Args:
            pc1: torch.Tensor, shape (B, N, D) - First batch of point clouds.
            pc2: torch.Tensor, shape (B, M, D) - Second batch of point clouds.

        Returns:
            squared_dist_matrix: torch.Tensor, shape (B, N, M) - Squared distance matrix for each batch.
        """
        diff = pc1[:, :, None, :] - pc2[:, None, :, :]  # Shape: (B, N, M, D)
        squared_dist_matrix = torch.sum(diff ** 2, dim=-1)  # Shape: (B, N, M)
        return squared_dist_matrix

    def __call__(
            self,
            pc1: torch.Tensor,
            pc2: torch.Tensor,
            bidirectional=False,
            batch_reduction="mean",
            point_reduction="sum"
    ):
        """
        Compute the Chamfer Distance between two sets of point clouds.

        Args:
            pc1: torch.Tensor, shape (B, N, 3) - First batch of point clouds.
            pc2: torch.Tensor, shape (B, M, 3) - Second batch of point clouds.
            bidirectional: bool - If True, calculate two-way Chamfer Distance. If False, calculate one-way.
            batch_reduction: str - Reduction over batch ('mean', 'sum', or None).
            point_reduction: str - Reduction over points ('sum' or 'mean').
        Returns:
            chamfer_dist: torch.Tensor - Chamfer Distance (scalar if batch_reduction is specified, else shape (B,))
        """
        # Ensure the inputs are 3D tensors
        assert pc1.ndim == 3 and pc2.ndim == 3, "Input point clouds must have shape (B, N, 3) and (B, M, 3)"
        assert pc1.shape[0] == pc2.shape[0], "Both inputs must have the same batch size"
        assert pc1.shape[2] == pc2.shape[2], "Point clouds must have the same dimensionality (C)"

        # Compute pairwise distance matrix: (B, N, M)
        if self.use_squared_cdist:
            dist_matrix = self.squared_cdist(pc1, pc2)
        else:
            dist_matrix = torch.cdist(pc1, pc2, p=2)  # Torch's cdist supports batch processing directly.

        # Compute minimum distances from pc1 to pc2
        min_dist_pc1_to_pc2 = torch.min(dist_matrix, dim=2)[0]  # Shape: (B, N)

        if not bidirectional:
            # One-way Chamfer Distance (from pc1 to pc2)
            if point_reduction == 'sum':
                chamfer_dist = torch.sum(min_dist_pc1_to_pc2, dim=1)  # Reduce over points, keep batch dimension
            elif point_reduction == 'mean':
                chamfer_dist = torch.mean(min_dist_pc1_to_pc2, dim=1)
            else:
                raise ValueError("Invalid point_reduction. Must be either 'sum' or 'mean'.")
        else:
            # Compute minimum distances from pc2 to pc1 for two-way Chamfer Distance
            min_dist_pc2_to_pc1 = torch.min(dist_matrix, dim=1)[0]  # Shape: (B, M)
            if point_reduction == 'sum':
                chamfer_dist = torch.sum(min_dist_pc1_to_pc2, dim=1) + torch.sum(min_dist_pc2_to_pc1, dim=1)
            elif point_reduction == 'mean':
                chamfer_dist = torch.mean(min_dist_pc1_to_pc2, dim=1) + torch.mean(min_dist_pc2_to_pc1, dim=1)
            else:
                raise ValueError("Invalid point_reduction. Must be either 'sum' or 'mean'.")

        # Batch reduction
        if batch_reduction == 'mean':
            chamfer_dist = torch.mean(chamfer_dist)  # Average over batch
        elif batch_reduction == 'sum':
            chamfer_dist = torch.sum(chamfer_dist)  # Sum over batch
        elif batch_reduction is None:
            pass  # Return per-batch Chamfer Distance without reduction
        else:
            raise ValueError("Invalid batch_reduction. Must be 'mean', 'sum', or None.")

        # Optional correctness check
        if CHECK_CORRECTNESS:
            from chamferdist import ChamferDistance as ChamferDistanceLib

            # Check the correctness of the Chamfer Distance
            chamfer_dist_off = ChamferDistanceLib()(pc1, pc2,
                                                    bidirectional=bidirectional,
                                                    batch_reduction=batch_reduction,
                                                    point_reduction=point_reduction)
            if batch_reduction is None:
                assert torch.allclose(chamfer_dist, chamfer_dist_off, atol=1e-6)
            else:
                assert torch.isclose(chamfer_dist, chamfer_dist_off, atol=1e-6)

        return chamfer_dist
