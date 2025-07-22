# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


def inverse_sigmoid(x, eps: float = 1e-5):
    x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
    x = ttnn.clamp(x, min=0, max=1)
    x1 = ttnn.clamp(x, min=eps)
    if len(x.shape) == 3:
        x_temp = ttnn.ones(shape=[x.shape[0], x.shape[1], x.shape[2]], layout=ttnn.TILE_LAYOUT, device=x.device())
    else:
        x_temp = ttnn.ones(
            shape=[x.shape[0], x.shape[1], x.shape[2], x.shape[3]], layout=ttnn.TILE_LAYOUT, device=x.device()
        )
    x_temp = x_temp - x
    x2 = ttnn.clamp(x_temp, min=eps)
    return ttnn.log(ttnn.div(x1, x2))


def bbox_xyxy_to_cxcywh(bbox):
    bbox = ttnn.unsqueeze(bbox, 0)
    bbox = ttnn.to_layout(bbox, layout=ttnn.ROW_MAJOR_LAYOUT)
    x1, y1, x2, y2 = ttnn.split(bbox, (1, 1, 1, 1), 2)
    bbox_new = [ttnn.div((x1 + x2), 2), ttnn.div((y1 + y2), 2), (x2 - x1), (y2 - y1)]
    return ttnn.concat(bbox_new, dim=-1)


def bbox_cxcywh_to_xyxy(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = ttnn.split(bbox, (1, 1, 1, 1), dim=-1)

    bbox_new = [ttnn.mul((cx - 0.5), w), ttnn.mul((cy - 0.5), h), ttnn.mul((cx + 0.5), w), ttnn.mul((cy + 0.5), h)]
    return ttnn.concat(bbox_new, dim=-1)


def tt_denormalize_2d_bbox(bboxes, pc_range):
    bboxes = bbox_cxcywh_to_xyxy(bboxes)

    bboxes_reshaped = ttnn.reshape(bboxes, (bboxes.shape[0], 2, 2))
    bboxes_even = bboxes_reshaped[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    bboxex_odd = bboxes_reshaped[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]

    bboxes_combined = ttnn.stack([bboxes_even, bboxex_odd], dim=-1)
    bboxes = ttnn.reshape(bboxes_combined, [bboxes.shape[0], -1])

    return bboxes


def tt_denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts_slice1 = new_pts[..., 0:1]
    new_pts_slice2 = new_pts[..., 1:2]
    new_pts_slice1 = pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    new_pts_slice2 = pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    new_pts = ttnn.concat([new_pts_slice1, new_pts_slice2], dim=-1)
    return new_pts
