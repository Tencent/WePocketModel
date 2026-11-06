"""Geometry helpers shared by Qwen3.5 True MGVB and multimodal M-RoPE."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, order=True)
class SpatialCell:
    temporal: int
    top: int
    left: int
    bottom: int
    right: int

    @property
    def area(self) -> int:
        return (self.bottom - self.top) * (self.right - self.left)

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left


def merged_grid_shape(grid_thw, spatial_merge_size: int) -> tuple[int, int, int]:
    values = grid_thw.detach().to(device="cpu", dtype=torch.long).tolist() if isinstance(grid_thw, torch.Tensor) else list(grid_thw)
    if len(values) != 3:
        raise ValueError(f"grid_thw must contain exactly three values, got {values!r}")
    temporal, height, width = (int(value) for value in values)
    merge = int(spatial_merge_size)
    if temporal < 1 or height < 1 or width < 1 or merge < 1:
        raise ValueError(f"invalid visual grid={values!r}, spatial_merge_size={merge}")
    if height % merge or width % merge:
        raise ValueError(f"visual grid {values!r} is not divisible by spatial_merge_size={merge}")
    return temporal, height // merge, width // merge


def native_visual_token_count(grid_thw, spatial_merge_size: int) -> int:
    temporal, height, width = merged_grid_shape(grid_thw, spatial_merge_size)
    return temporal * height * width


def compact_visual_token_count(grid_thw, spatial_merge_size: int, token_budget: int) -> int:
    budget = int(token_budget)
    if budget < 1:
        raise ValueError(f"MGVB token budget must be positive, got {budget}")
    return min(native_visual_token_count(grid_thw, spatial_merge_size), budget)


def native_visual_positions(grid_thw, spatial_merge_size: int, *, device=None) -> torch.Tensor:
    temporal, height, width = merged_grid_shape(grid_thw, spatial_merge_size)
    t = torch.arange(temporal, device=device, dtype=torch.long).view(-1, 1, 1)
    h = torch.arange(height, device=device, dtype=torch.long).view(1, -1, 1)
    w = torch.arange(width, device=device, dtype=torch.long).view(1, 1, -1)
    return torch.stack(
        [
            t.expand(temporal, height, width),
            h.expand(temporal, height, width),
            w.expand(temporal, height, width),
        ],
        dim=-1,
    ).reshape(-1, 3)


def partition_spatial_grid(
    temporal: int,
    height: int,
    width: int,
    cell_count: int,
) -> list[SpatialCell]:
    """Split a THW grid into exactly ``cell_count`` deterministic non-empty rectangles."""

    temporal, height, width, cell_count = map(int, (temporal, height, width, cell_count))
    token_count = temporal * height * width
    if temporal < 1 or height < 1 or width < 1:
        raise ValueError(f"invalid merged grid {(temporal, height, width)!r}")
    if cell_count < 1 or cell_count > token_count:
        raise ValueError(f"cell_count={cell_count} must be in [1, {token_count}]")

    cells = [SpatialCell(t, 0, 0, height, width) for t in range(temporal)]
    if cell_count < len(cells):
        raise ValueError(
            f"cell_count={cell_count} cannot preserve temporal slices for temporal={temporal}"
        )

    while len(cells) < cell_count:
        candidates = [
            (cell.area, max(cell.height, cell.width), -index, index)
            for index, cell in enumerate(cells)
            if cell.area > 1
        ]
        if not candidates:
            raise RuntimeError(
                f"cannot split merged grid {(temporal, height, width)} into {cell_count} cells"
            )
        _, _, _, split_index = max(candidates)
        cell = cells.pop(split_index)
        if cell.width >= cell.height and cell.width > 1:
            middle = cell.left + cell.width // 2
            children = [
                SpatialCell(cell.temporal, cell.top, cell.left, cell.bottom, middle),
                SpatialCell(cell.temporal, cell.top, middle, cell.bottom, cell.right),
            ]
        else:
            middle = cell.top + cell.height // 2
            children = [
                SpatialCell(cell.temporal, cell.top, cell.left, middle, cell.right),
                SpatialCell(cell.temporal, middle, cell.left, cell.bottom, cell.right),
            ]
        cells.extend(children)

    return sorted(cells)


def cell_center_positions(cells: list[SpatialCell], *, device=None) -> torch.Tensor:
    return torch.tensor(
        [
            [
                cell.temporal,
                (cell.top + cell.bottom - 1) // 2,
                (cell.left + cell.right - 1) // 2,
            ]
            for cell in cells
        ],
        dtype=torch.long,
        device=device,
    )


def cell_edges(
    cells: list[SpatialCell],
    height: int,
    width: int,
    *,
    halo: int = 0,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened token indices and their destination cell ids."""

    token_indices: list[int] = []
    cell_indices: list[int] = []
    halo = max(0, int(halo))
    for cell_id, cell in enumerate(cells):
        top = max(0, cell.top - halo)
        left = max(0, cell.left - halo)
        bottom = min(height, cell.bottom + halo)
        right = min(width, cell.right + halo)
        base = cell.temporal * height * width
        for row in range(top, bottom):
            start = base + row * width + left
            count = right - left
            token_indices.extend(range(start, start + count))
            cell_indices.extend([cell_id] * count)
    return (
        torch.tensor(token_indices, dtype=torch.long, device=device),
        torch.tensor(cell_indices, dtype=torch.long, device=device),
    )


def cell_assignment(
    cells: list[SpatialCell],
    height: int,
    width: int,
    *,
    device=None,
) -> torch.Tensor:
    if not cells:
        raise ValueError("cannot build a cell assignment from an empty cell list")
    temporal = max(cell.temporal for cell in cells) + 1
    assignment = torch.full(
        (temporal, height, width), -1, dtype=torch.long, device=device
    )
    for cell_id, cell in enumerate(cells):
        assignment[cell.temporal, cell.top : cell.bottom, cell.left : cell.right] = cell_id
    if torch.any(assignment < 0):
        raise RuntimeError("spatial cells do not cover the complete visual grid")
    return assignment.reshape(-1)


def build_mgvb_visual_positions(
    grid_thw,
    *,
    spatial_merge_size: int,
    token_budget: int,
    global_tokens: int,
    region_tokens: int,
    device=None,
) -> torch.Tensor:
    """Build positions in the exact order emitted by True MGVB."""

    temporal, height, width = merged_grid_shape(grid_thw, spatial_merge_size)
    native_count = temporal * height * width
    compact_count = min(native_count, int(token_budget))
    if compact_count == native_count:
        return native_visual_positions(grid_thw, spatial_merge_size, device=device)

    global_count = int(global_tokens)
    region_count = int(region_tokens)
    fine_count = compact_count - global_count - region_count
    if global_count < 1 or region_count < 1 or fine_count < 1:
        raise ValueError(
            "compressed MGVB budget must leave positive global, region, and fine counts; "
            f"got total={compact_count}, global={global_count}, region={region_count}, fine={fine_count}"
        )

    center = torch.tensor(
        [[(temporal - 1) // 2, (height - 1) // 2, (width - 1) // 2]],
        dtype=torch.long,
        device=device,
    ).expand(global_count, -1)
    region_cells = partition_spatial_grid(temporal, height, width, region_count)
    fine_cells = partition_spatial_grid(temporal, height, width, fine_count)
    return torch.cat(
        [
            center,
            cell_center_positions(region_cells, device=device),
            cell_center_positions(fine_cells, device=device),
        ],
        dim=0,
    )
