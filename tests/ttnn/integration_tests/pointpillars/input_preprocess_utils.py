from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
import mmengine
import warnings
from mmengine.fileio import get
from pathlib import Path

# from mmdet3d.datasets.transforms.loading.py import LoadMultiViewImageFromFiles
# from mmdet3d.datasets.transforms import  LidarDet3DInferencerLoader
# from models.experimental.functional_pointpillars.reference.point_pillars_utils import
from mmcv.transforms import BaseTransform
import torch
from abc import abstractmethod
import functools
from torch import Tensor
from typing import Iterator, Optional, Sequence, Union, Dict, Type
from models.experimental.functional_pointpillars.reference.point_pillars_utils import BaseInstance3DBoxes
from copy import deepcopy
from enum import IntEnum, unique
import mmcv

# from mmdet.datasets.transforms import RandomFlip


class Box3DMode(IntEnum):
    """Enum of different ways to represent a box.

    Coordinates in LiDAR:

    .. code-block:: none

                    up z
                       ^   x front
                       |  /
                       | /
        left y <------ 0

    The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.

    Coordinates in Camera:

    .. code-block:: none

                z front
               /
              /
             0 ------> x right
             |
             |
             v
        down y

    The relative coordinate of bottom center in a CAM box is (0.5, 1.0, 0.5),
    and the yaw is around the y axis, thus the rotation axis=1.

    Coordinates in Depth:

    .. code-block:: none

        up z
           ^   y front
           |  /
           | /
           0 ------> x right

    The relative coordinate of bottom center in a DEPTH box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    """

    LIDAR = 0
    CAM = 1
    DEPTH = 2

    @staticmethod
    def convert(
        box: Union[Sequence[float], np.ndarray, Tensor, BaseInstance3DBoxes],
        src: "Box3DMode",
        dst: "Box3DMode",
        rt_mat: Optional[Union[np.ndarray, Tensor]] = None,
        with_yaw: bool = True,
        correct_yaw: bool = False,
    ) -> Union[Sequence[float], np.ndarray, Tensor, BaseInstance3DBoxes]:
        """Convert boxes from ``src`` mode to ``dst`` mode.

        Args:
            box (Sequence[float] or np.ndarray or Tensor or
                :obj:`BaseInstance3DBoxes`): Can be a k-tuple, k-list or an Nxk
                array/tensor.
            src (:obj:`Box3DMode`): The source box mode.
            dst (:obj:`Box3DMode`): The target box mode.
            rt_mat (np.ndarray or Tensor, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.
            with_yaw (bool): If ``box`` is an instance of
                :obj:`BaseInstance3DBoxes`, whether or not it has a yaw angle.
                Defaults to True.
            correct_yaw (bool): If the yaw is rotated by rt_mat.
                Defaults to False.

        Returns:
            Sequence[float] or np.ndarray or Tensor or
            :obj:`BaseInstance3DBoxes`: The converted box of the same type.
        """
        if src == dst:
            return box

        is_numpy = isinstance(box, np.ndarray)
        is_Instance3DBoxes = isinstance(box, BaseInstance3DBoxes)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) >= 7, (
                "Box3DMode.convert takes either a k-tuple/list or " "an Nxk array/tensor, where k >= 7"
            )
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            elif is_Instance3DBoxes:
                arr = box.tensor.clone()
            else:
                arr = box.clone()

        if is_Instance3DBoxes:
            with_yaw = box.with_yaw

        # convert box from `src` mode to `dst` mode.
        x_size, y_size, z_size = arr[..., 3:4], arr[..., 4:5], arr[..., 5:6]
        if with_yaw:
            yaw = arr[..., 6:7]
        if src == Box3DMode.LIDAR and dst == Box3DMode.CAM:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
            if with_yaw:
                if correct_yaw:
                    yaw_vector = torch.cat([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=1)
                else:
                    yaw = -yaw - np.pi / 2
                    yaw = limit_period(yaw, period=np.pi * 2)
        elif src == Box3DMode.CAM and dst == Box3DMode.LIDAR:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
            if with_yaw:
                if correct_yaw:
                    yaw_vector = torch.cat([torch.cos(-yaw), torch.zeros_like(yaw), torch.sin(-yaw)], dim=1)
                else:
                    yaw = -yaw - np.pi / 2
                    yaw = limit_period(yaw, period=np.pi * 2)
        elif src == Box3DMode.DEPTH and dst == Box3DMode.CAM:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
            if with_yaw:
                if correct_yaw:
                    yaw_vector = torch.cat([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=1)
                else:
                    yaw = -yaw
        elif src == Box3DMode.CAM and dst == Box3DMode.DEPTH:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
            if with_yaw:
                if correct_yaw:
                    yaw_vector = torch.cat([torch.cos(-yaw), torch.zeros_like(yaw), torch.sin(-yaw)], dim=1)
                else:
                    yaw = -yaw
        elif src == Box3DMode.LIDAR and dst == Box3DMode.DEPTH:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            xyz_size = torch.cat([x_size, y_size, z_size], dim=-1)
            if with_yaw:
                if correct_yaw:
                    yaw_vector = torch.cat([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=1)
                else:
                    yaw = yaw + np.pi / 2
                    yaw = limit_period(yaw, period=np.pi * 2)
        elif src == Box3DMode.DEPTH and dst == Box3DMode.LIDAR:
            if rt_mat is None:
                rt_mat = arr.new_tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            xyz_size = torch.cat([x_size, y_size, z_size], dim=-1)
            if with_yaw:
                if correct_yaw:
                    yaw_vector = torch.cat([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=1)
                else:
                    yaw = yaw - np.pi / 2
                    yaw = limit_period(yaw, period=np.pi * 2)
        else:
            raise NotImplementedError(f"Conversion from Box3DMode {src} to {dst} " "is not supported yet")

        if not isinstance(rt_mat, Tensor):
            rt_mat = arr.new_tensor(rt_mat)
        if rt_mat.size(1) == 4:
            extended_xyz = torch.cat([arr[..., :3], arr.new_ones(arr.size(0), 1)], dim=-1)
            xyz = extended_xyz @ rt_mat.t()
        else:
            xyz = arr[..., :3] @ rt_mat.t()

        # Note: we only use rotation in rt_mat
        # so don't need to extend yaw_vector
        if with_yaw and correct_yaw:
            rot_yaw_vector = yaw_vector @ rt_mat[:3, :3].t()
            if dst == Box3DMode.CAM:
                yaw = torch.atan2(-rot_yaw_vector[:, [2]], rot_yaw_vector[:, [0]])
            elif dst in [Box3DMode.LIDAR, Box3DMode.DEPTH]:
                yaw = torch.atan2(rot_yaw_vector[:, [1]], rot_yaw_vector[:, [0]])
            yaw = limit_period(yaw, period=np.pi * 2)

        if with_yaw:
            remains = arr[..., 7:]
            arr = torch.cat([xyz[..., :3], xyz_size, yaw, remains], dim=-1)
        else:
            remains = arr[..., 6:]
            arr = torch.cat([xyz[..., :3], xyz_size, remains], dim=-1)

        # convert arr to the original type
        original_type = type(box)
        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        elif is_Instance3DBoxes:
            if dst == Box3DMode.CAM:
                target_type = CameraInstance3DBoxes
            elif dst == Box3DMode.LIDAR:
                target_type = LiDARInstance3DBoxes
            elif dst == Box3DMode.DEPTH:
                target_type = DepthInstance3DBoxes
            else:
                raise NotImplementedError(f"Conversion to {dst} through {original_type} " "is not supported yet")
            return target_type(arr, box_dim=arr.size(-1), with_yaw=with_yaw)
        else:
            return arr


from inspect import getfullargspec

TemplateArrayType = Union[np.ndarray, torch.Tensor, list, tuple, int, float]


class ArrayConverter:
    """Utility class for data-type agnostic processing.

    Args:
        template_array (np.ndarray or torch.Tensor or list or tuple or int or
            float, optional): Template array. Defaults to None.
    """

    SUPPORTED_NON_ARRAY_TYPES = (
        int,
        float,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
    )

    def __init__(self, template_array: Optional[TemplateArrayType] = None) -> None:
        if template_array is not None:
            self.set_template(template_array)

    def set_template(self, array: TemplateArrayType) -> None:
        """Set template array.

        Args:
            array (np.ndarray or torch.Tensor or list or tuple or int or
                float): Template array.

        Raises:
            ValueError: If input is list or tuple and cannot be converted to a
                NumPy array, a ValueError is raised.
            TypeError: If input type does not belong to the above range, or the
                contents of a list or tuple do not share the same data type, a
                TypeError is raised.
        """
        self.array_type = type(array)
        self.is_num = False
        self.device = "cpu"

        if isinstance(array, np.ndarray):
            self.dtype = array.dtype
        elif isinstance(array, torch.Tensor):
            self.dtype = array.dtype
            self.device = array.device
        elif isinstance(array, (list, tuple)):
            try:
                array = np.array(array)
                if array.dtype not in self.SUPPORTED_NON_ARRAY_TYPES:
                    raise TypeError
                self.dtype = array.dtype
            except (ValueError, TypeError):
                print("The following list cannot be converted to a numpy " f"array of supported dtype:\n{array}")
                raise
        elif isinstance(array, (int, float)):
            self.array_type = np.ndarray
            self.is_num = True
            self.dtype = np.dtype(type(array))
        else:
            raise TypeError(f"Template type {self.array_type} is not supported.")

    def convert(
        self,
        input_array: TemplateArrayType,
        target_type: Optional[Type] = None,
        target_array: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Convert input array to target data type.

        Args:
            input_array (np.ndarray or torch.Tensor or list or tuple or int or
                float): Input array.
            target_type (Type, optional): Type to which input array is
                converted. It should be `np.ndarray` or `torch.Tensor`.
                Defaults to None.
            target_array (np.ndarray or torch.Tensor, optional): Template array
                to which input array is converted. Defaults to None.

        Raises:
            ValueError: If input is list or tuple and cannot be converted to a
                NumPy array, a ValueError is raised.
            TypeError: If input type does not belong to the above range, or the
                contents of a list or tuple do not share the same data type, a
                TypeError is raised.

        Returns:
            np.ndarray or torch.Tensor: The converted array.
        """
        if isinstance(input_array, (list, tuple)):
            try:
                input_array = np.array(input_array)
                if input_array.dtype not in self.SUPPORTED_NON_ARRAY_TYPES:
                    raise TypeError
            except (ValueError, TypeError):
                print("The input cannot be converted to a single-type numpy " f"array:\n{input_array}")
                raise
        elif isinstance(input_array, self.SUPPORTED_NON_ARRAY_TYPES):
            input_array = np.array(input_array)
        array_type = type(input_array)
        assert target_type is not None or target_array is not None, "must specify a target"
        if target_type is not None:
            assert target_type in (np.ndarray, torch.Tensor), "invalid target type"
            if target_type == array_type:
                return input_array
            elif target_type == np.ndarray:
                # default dtype is float32
                converted_array = input_array.cpu().numpy().astype(np.float32)
            else:
                # default dtype is float32, device is 'cpu'
                converted_array = torch.tensor(input_array, dtype=torch.float32)
        else:
            assert isinstance(target_array, (np.ndarray, torch.Tensor)), "invalid target array type"
            if isinstance(target_array, array_type):
                return input_array
            elif isinstance(target_array, np.ndarray):
                converted_array = input_array.cpu().numpy().astype(target_array.dtype)
            else:
                converted_array = target_array.new_tensor(input_array)
        return converted_array

    def recover(self, input_array: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor, int, float]:
        """Recover input type to original array type.

        Args:
            input_array (np.ndarray or torch.Tensor): Input array.

        Returns:
            np.ndarray or torch.Tensor or int or float: Converted array.
        """
        assert isinstance(input_array, (np.ndarray, torch.Tensor)), "invalid input array type"
        if isinstance(input_array, self.array_type):
            return input_array
        elif isinstance(input_array, torch.Tensor):
            converted_array = input_array.cpu().numpy().astype(self.dtype)
        else:
            converted_array = torch.tensor(input_array, dtype=self.dtype, device=self.device)
        if self.is_num:
            converted_array = converted_array.item()
        return converted_array


def array_converter(
    to_torch: bool = True,
    apply_to: Tuple[str, ...] = tuple(),
    template_arg_name_: Optional[str] = None,
    recover: bool = True,
) -> Callable:
    """Wrapper function for data-type agnostic processing.

    First converts input arrays to PyTorch tensors or NumPy arrays for middle
    calculation, then convert output to original data-type if `recover=True`.

    Args:
        to_torch (bool): Whether to convert to PyTorch tensors for middle
            calculation. Defaults to True.
        apply_to (Tuple[str]): The arguments to which we apply data-type
            conversion. Defaults to an empty tuple.
        template_arg_name_ (str, optional): Argument serving as the template
            (return arrays should have the same dtype and device as the
            template). Defaults to None. If None, we will use the first
            argument in `apply_to` as the template argument.
        recover (bool): Whether or not to recover the wrapped function outputs
            to the `template_arg_name_` type. Defaults to True.

    Raises:
        ValueError: When template_arg_name_ is not among all args, or when
            apply_to contains an arg which is not among all args, a ValueError
            will be raised. When the template argument or an argument to
            convert is a list or tuple, and cannot be converted to a NumPy
            array, a ValueError will be raised.
        TypeError: When the type of the template argument or an argument to
            convert does not belong to the above range, or the contents of such
            an list-or-tuple-type argument do not share the same data type, a
            TypeError will be raised.

    Returns:
        Callable: Wrapped function.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>>
        >>> # Use torch addition for a + b,
        >>> # and convert return values to the type of a
        >>> @array_converter(apply_to=('a', 'b'))
        >>> def simple_add(a, b):
        >>>     return a + b
        >>>
        >>> a = np.array([1.1])
        >>> b = np.array([2.2])
        >>> simple_add(a, b)
        >>>
        >>> # Use numpy addition for a + b,
        >>> # and convert return values to the type of b
        >>> @array_converter(to_torch=False, apply_to=('a', 'b'),
        >>>                  template_arg_name_='b')
        >>> def simple_add(a, b):
        >>>     return a + b
        >>>
        >>> simple_add(a, b)
        >>>
        >>> # Use torch funcs for floor(a) if flag=True else ceil(a),
        >>> # and return the torch tensor
        >>> @array_converter(apply_to=('a',), recover=False)
        >>> def floor_or_ceil(a, flag=True):
        >>>     return torch.floor(a) if flag else torch.ceil(a)
        >>>
        >>> floor_or_ceil(a, flag=False)
    """

    def array_converter_wrapper(func):
        """Outer wrapper for the function."""

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            """Inner wrapper for the arguments."""
            if len(apply_to) == 0:
                return func(*args, **kwargs)

            func_name = func.__name__

            arg_spec = getfullargspec(func)

            arg_names = arg_spec.args
            arg_num = len(arg_names)
            default_arg_values = arg_spec.defaults
            if default_arg_values is None:
                default_arg_values = []
            no_default_arg_num = len(arg_names) - len(default_arg_values)

            kwonly_arg_names = arg_spec.kwonlyargs
            kwonly_default_arg_values = arg_spec.kwonlydefaults
            if kwonly_default_arg_values is None:
                kwonly_default_arg_values = {}

            all_arg_names = arg_names + kwonly_arg_names

            # in case there are args in the form of *args
            if len(args) > arg_num:
                named_args = args[:arg_num]
                nameless_args = args[arg_num:]
            else:
                named_args = args
                nameless_args = []

            # template argument data type is used for all array-like arguments
            if template_arg_name_ is None:
                template_arg_name = apply_to[0]
            else:
                template_arg_name = template_arg_name_

            if template_arg_name not in all_arg_names:
                raise ValueError(f"{template_arg_name} is not among the " f"argument list of function {func_name}")

            # inspect apply_to
            for arg_to_apply in apply_to:
                if arg_to_apply not in all_arg_names:
                    raise ValueError(f"{arg_to_apply} is not an argument of {func_name}")

            new_args = []
            new_kwargs = {}

            converter = ArrayConverter()
            target_type = torch.Tensor if to_torch else np.ndarray

            # non-keyword arguments
            for i, arg_value in enumerate(named_args):
                if arg_names[i] in apply_to:
                    new_args.append(converter.convert(input_array=arg_value, target_type=target_type))
                else:
                    new_args.append(arg_value)

                if arg_names[i] == template_arg_name:
                    template_arg_value = arg_value

            kwonly_default_arg_values.update(kwargs)
            kwargs = kwonly_default_arg_values

            # keyword arguments and non-keyword arguments using default value
            for i in range(len(named_args), len(all_arg_names)):
                arg_name = all_arg_names[i]
                if arg_name in kwargs:
                    if arg_name in apply_to:
                        new_kwargs[arg_name] = converter.convert(input_array=kwargs[arg_name], target_type=target_type)
                    else:
                        new_kwargs[arg_name] = kwargs[arg_name]
                else:
                    default_value = default_arg_values[i - no_default_arg_num]
                    if arg_name in apply_to:
                        new_kwargs[arg_name] = converter.convert(input_array=default_value, target_type=target_type)
                    else:
                        new_kwargs[arg_name] = default_value
                if arg_name == template_arg_name:
                    template_arg_value = kwargs[arg_name]

            # add nameless args provided by *args (if exists)
            new_args += nameless_args

            return_values = func(*new_args, **new_kwargs)
            converter.set_template(template_arg_value)

            def recursive_recover(input_data):
                if isinstance(input_data, (tuple, list)):
                    new_data = []
                    for item in input_data:
                        new_data.append(recursive_recover(item))
                    return tuple(new_data) if isinstance(input_data, tuple) else new_data
                elif isinstance(input_data, dict):
                    new_data = {}
                    for k, v in input_data.items():
                        new_data[k] = recursive_recover(v)
                    return new_data
                elif isinstance(input_data, (torch.Tensor, np.ndarray)):
                    return converter.recover(input_data)
                else:
                    return input_data

            if recover:
                return recursive_recover(return_values)
            else:
                return return_values

        return new_func

    return array_converter_wrapper


@array_converter(apply_to=("points", "angles"))
def rotation_3d_in_axis(
    points: Union[np.ndarray, Tensor],
    angles: Union[np.ndarray, Tensor, float],
    axis: int = 0,
    return_mat: bool = False,
    clockwise: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[Tensor, Tensor], np.ndarray, Tensor]:
    """Rotate points by angles according to axis.

    Args:
        points (np.ndarray or Tensor): Points with shape (N, M, 3).
        angles (np.ndarray or Tensor or float): Vector of angles with shape
            (N, ).
        axis (int): The axis to be rotated. Defaults to 0.
        return_mat (bool): Whether or not to return the rotation matrix
            (transposed). Defaults to False.
        clockwise (bool): Whether the rotation is clockwise. Defaults to False.

    Raises:
        ValueError: When the axis is not in range [-3, -2, -1, 0, 1, 2], it
            will raise ValueError.

    Returns:
        Tuple[np.ndarray, np.ndarray] or Tuple[Tensor, Tensor] or np.ndarray or
        Tensor: Rotated points with shape (N, M, 3) and rotation matrix with
        shape (N, 3, 3).
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if isinstance(angles, float) or len(angles.shape) == 0:
        angles = torch.full(points.shape[:1], angles)

    assert len(points.shape) == 3 and len(angles.shape) == 1 and points.shape[0] == angles.shape[0], (
        "Incorrect shape of points " f"angles: {points.shape}, {angles.shape}"
    )

    assert points.shape[-1] in [2, 3], f"Points size should be 2 or 3 instead of {points.shape[-1]}"

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    if points.shape[-1] == 3:
        if axis == 1 or axis == -2:
            rot_mat_T = torch.stack(
                [
                    torch.stack([rot_cos, zeros, -rot_sin]),
                    torch.stack([zeros, ones, zeros]),
                    torch.stack([rot_sin, zeros, rot_cos]),
                ]
            )
        elif axis == 2 or axis == -1:
            rot_mat_T = torch.stack(
                [
                    torch.stack([rot_cos, rot_sin, zeros]),
                    torch.stack([-rot_sin, rot_cos, zeros]),
                    torch.stack([zeros, zeros, ones]),
                ]
            )
        elif axis == 0 or axis == -3:
            rot_mat_T = torch.stack(
                [
                    torch.stack([ones, zeros, zeros]),
                    torch.stack([zeros, rot_cos, rot_sin]),
                    torch.stack([zeros, -rot_sin, rot_cos]),
                ]
            )
        else:
            raise ValueError(f"axis should in range [-3, -2, -1, 0, 1, 2], got {axis}")
    else:
        rot_mat_T = torch.stack([torch.stack([rot_cos, rot_sin]), torch.stack([-rot_sin, rot_cos])])

    if clockwise:
        rot_mat_T = rot_mat_T.transpose(0, 1)

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = torch.einsum("aij,jka->aik", points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        rot_mat_T = torch.einsum("jka->ajk", rot_mat_T)
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    else:
        return points_new


class BasePoints:
    """Base class for Points.

    Args:
        tensor (Tensor or np.ndarray or Sequence[Sequence[float]]): The points
            data with shape (N, points_dim).
        points_dim (int): Integer indicating the dimension of a point. Each row
            is (x, y, z, ...). Defaults to 3.
        attribute_dims (dict, optional): Dictionary to indicate the meaning of
            extra dimension. Defaults to None.

    Attributes:
        tensor (Tensor): Float matrix with shape (N, points_dim).
        points_dim (int): Integer indicating the dimension of a point. Each row
            is (x, y, z, ...).
        attribute_dims (dict, optional): Dictionary to indicate the meaning of
            extra dimension. Defaults to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(
        self,
        tensor: Union[Tensor, np.ndarray, Sequence[Sequence[float]]],
        points_dim: int = 3,
        attribute_dims: Optional[dict] = None,
    ) -> None:
        if isinstance(tensor, Tensor):
            device = tensor.device
        else:
            device = torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does
            # not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, points_dim))
        assert tensor.dim() == 2 and tensor.size(-1) == points_dim, (
            "The points dimension must be 2 and the length of the last "
            f"dimension must be {points_dim}, but got points with shape "
            f"{tensor.shape}."
        )

        self.tensor = tensor.clone()
        self.points_dim = points_dim
        self.attribute_dims = attribute_dims
        self.rotation_axis = 0

    @property
    def coord(self) -> Tensor:
        """Tensor: Coordinates of each point in shape (N, 3)."""
        return self.tensor[:, :3]

    @coord.setter
    def coord(self, tensor: Union[Tensor, np.ndarray]) -> None:
        """Set the coordinates of each point.

        Args:
            tensor (Tensor or np.ndarray): Coordinates of each point with shape
                (N, 3).
        """
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
            raise ValueError(f"got unexpected shape {tensor.shape}")
        if not isinstance(tensor, Tensor):
            tensor = self.tensor.new_tensor(tensor)
        self.tensor[:, :3] = tensor

    @property
    def height(self) -> Union[Tensor, None]:
        """Tensor or None: Returns a vector with height of each point in shape
        (N, )."""
        if self.attribute_dims is not None and "height" in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims["height"]]
        else:
            return None

    @height.setter
    def height(self, tensor: Union[Tensor, np.ndarray]) -> None:
        """Set the height of each point.

        Args:
            tensor (Tensor or np.ndarray): Height of each point with shape
                (N, ).
        """
        try:
            tensor = tensor.reshape(self.shape[0])
        except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
            raise ValueError(f"got unexpected shape {tensor.shape}")
        if not isinstance(tensor, Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and "height" in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims["height"]] = tensor
        else:
            # add height attribute
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor.unsqueeze(1)], dim=1)
            self.attribute_dims.update(dict(height=attr_dim))
            self.points_dim += 1

    @property
    def color(self) -> Union[Tensor, None]:
        """Tensor or None: Returns a vector with color of each point in shape
        (N, 3)."""
        if self.attribute_dims is not None and "color" in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims["color"]]
        else:
            return None

    @color.setter
    def color(self, tensor: Union[Tensor, np.ndarray]) -> None:
        """Set the color of each point.

        Args:
            tensor (Tensor or np.ndarray): Color of each point with shape
                (N, 3).
        """
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
            raise ValueError(f"got unexpected shape {tensor.shape}")
        if tensor.max() >= 256 or tensor.min() < 0:
            warnings.warn("point got color value beyond [0, 255]")
        if not isinstance(tensor, Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and "color" in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims["color"]] = tensor
        else:
            # add color attribute
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor], dim=1)
            self.attribute_dims.update(dict(color=[attr_dim, attr_dim + 1, attr_dim + 2]))
            self.points_dim += 3

    @property
    def shape(self) -> torch.Size:
        """torch.Size: Shape of points."""
        return self.tensor.shape

    def shuffle(self) -> Tensor:
        """Shuffle the points.

        Returns:
            Tensor: The shuffled index.
        """
        idx = torch.randperm(self.__len__(), device=self.tensor.device)
        self.tensor = self.tensor[idx]
        return idx

    def rotate(self, rotation: Union[Tensor, np.ndarray, float], axis: Optional[int] = None) -> Tensor:
        """Rotate points with the given rotation matrix or angle.

        Args:
            rotation (Tensor or np.ndarray or float): Rotation matrix or angle.
            axis (int, optional): Axis to rotate at. Defaults to None.

        Returns:
            Tensor: Rotation matrix.
        """
        if not isinstance(rotation, Tensor):
            rotation = self.tensor.new_tensor(rotation)
        assert rotation.shape == torch.Size([3, 3]) or rotation.numel() == 1, f"invalid rotation shape {rotation.shape}"

        if axis is None:
            axis = self.rotation_axis

        if rotation.numel() == 1:
            rotated_points, rot_mat_T = rotation_3d_in_axis(
                self.tensor[:, :3][None], rotation, axis=axis, return_mat=True
            )
            self.tensor[:, :3] = rotated_points.squeeze(0)
            rot_mat_T = rot_mat_T.squeeze(0)
        else:
            # rotation.numel() == 9
            self.tensor[:, :3] = self.tensor[:, :3] @ rotation
            rot_mat_T = rotation

        return rot_mat_T

    @abstractmethod
    def flip(self, bev_direction: str = "horizontal") -> None:
        """Flip the points along given BEV direction.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
                Defaults to 'horizontal'.
        """
        pass

    def translate(self, trans_vector: Union[Tensor, np.ndarray]) -> None:
        """Translate points with the given translation vector.

        Args:
            trans_vector (Tensor or np.ndarray): Translation vector of size 3
                or nx3.
        """
        if not isinstance(trans_vector, Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        trans_vector = trans_vector.squeeze(0)
        if trans_vector.dim() == 1:
            assert trans_vector.shape[0] == 3
        elif trans_vector.dim() == 2:
            assert trans_vector.shape[0] == self.tensor.shape[0] and trans_vector.shape[1] == 3
        else:
            raise NotImplementedError(f"Unsupported translation vector of shape {trans_vector.shape}")
        self.tensor[:, :3] += trans_vector

    def in_range_3d(self, point_range: Union[Tensor, np.ndarray, Sequence[float]]) -> Tensor:
        """Check whether the points are in the given range.

        Args:
            point_range (Tensor or np.ndarray or Sequence[float]): The range of
                point (x_min, y_min, z_min, x_max, y_max, z_max).

        Note:
            In the original implementation of SECOND, checking whether a box in
            the range checks whether the points are in a convex polygon, we try
            to reduce the burden for simpler cases.

        Returns:
            Tensor: A binary vector indicating whether each point is inside the
            reference range.
        """
        in_range_flags = (
            (self.tensor[:, 0] > point_range[0])
            & (self.tensor[:, 1] > point_range[1])
            & (self.tensor[:, 2] > point_range[2])
            & (self.tensor[:, 0] < point_range[3])
            & (self.tensor[:, 1] < point_range[4])
            & (self.tensor[:, 2] < point_range[5])
        )
        return in_range_flags

    @property
    def bev(self) -> Tensor:
        """Tensor: BEV of the points in shape (N, 2)."""
        return self.tensor[:, [0, 1]]

    def in_range_bev(self, point_range: Union[Tensor, np.ndarray, Sequence[float]]) -> Tensor:
        """Check whether the points are in the given range.

        Args:
            point_range (Tensor or np.ndarray or Sequence[float]): The range of
                point in order of (x_min, y_min, x_max, y_max).

        Returns:
            Tensor: A binary vector indicating whether each point is inside the
            reference range.
        """
        in_range_flags = (
            (self.bev[:, 0] > point_range[0])
            & (self.bev[:, 1] > point_range[1])
            & (self.bev[:, 0] < point_range[2])
            & (self.bev[:, 1] < point_range[3])
        )
        return in_range_flags

    @abstractmethod
    def convert_to(self, dst: int, rt_mat: Optional[Union[Tensor, np.ndarray]] = None) -> "BasePoints":
        """Convert self to ``dst`` mode.

        Args:
            dst (int): The target Point mode.
            rt_mat (Tensor or np.ndarray, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.

        Returns:
            :obj:`BasePoints`: The converted point of the same type in the
            ``dst`` mode.
        """
        pass

    def scale(self, scale_factor: float) -> None:
        """Scale the points with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the points.
        """
        self.tensor[:, :3] *= scale_factor

    def __getitem__(self, item: Union[int, tuple, slice, np.ndarray, Tensor]) -> "BasePoints":
        """
        Args:
            item (int or tuple or slice or np.ndarray or Tensor): Index of
                points.

        Note:
            The following usage are allowed:

            1. `new_points = points[3]`: Return a `Points` that contains only
               one point.
            2. `new_points = points[2:10]`: Return a slice of points.
            3. `new_points = points[vector]`: Whether vector is a
               torch.BoolTensor with `length = len(points)`. Nonzero elements
               in the vector will be selected.
            4. `new_points = points[3:11, vector]`: Return a slice of points
               and attribute dims.
            5. `new_points = points[4:12, 2]`: Return a slice of points with
               single attribute.

            Note that the returned Points might share storage with this Points,
            subject to PyTorch's indexing semantics.

        Returns:
            :obj:`BasePoints`: A new object of :class:`BasePoints` after
            indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(
                self.tensor[item].view(1, -1), points_dim=self.points_dim, attribute_dims=self.attribute_dims
            )
        elif isinstance(item, tuple) and len(item) == 2:
            if isinstance(item[1], slice):
                start = 0 if item[1].start is None else item[1].start
                stop = self.tensor.shape[1] if item[1].stop is None else item[1].stop
                step = 1 if item[1].step is None else item[1].step
                item = list(item)
                item[1] = list(range(start, stop, step))
                item = tuple(item)
            elif isinstance(item[1], int):
                item = list(item)
                item[1] = [item[1]]
                item = tuple(item)
            p = self.tensor[item[0], item[1]]

            keep_dims = list(set(item[1]).intersection(set(range(3, self.tensor.shape[1]))))
            if self.attribute_dims is not None:
                attribute_dims = self.attribute_dims.copy()
                for key in self.attribute_dims.keys():
                    cur_attribute_dims = attribute_dims[key]
                    if isinstance(cur_attribute_dims, int):
                        cur_attribute_dims = [cur_attribute_dims]
                    intersect_attr = list(set(cur_attribute_dims).intersection(set(keep_dims)))
                    if len(intersect_attr) == 1:
                        attribute_dims[key] = intersect_attr[0]
                    elif len(intersect_attr) > 1:
                        attribute_dims[key] = intersect_attr
                    else:
                        attribute_dims.pop(key)
            else:
                attribute_dims = None
        elif isinstance(item, (slice, np.ndarray, Tensor)):
            p = self.tensor[item]
            attribute_dims = self.attribute_dims
        else:
            raise NotImplementedError(f"Invalid slice {item}!")

        assert p.dim() == 2, f"Indexing on Points with {item} failed to return a matrix!"
        return original_type(p, points_dim=p.shape[1], attribute_dims=attribute_dims)

    def __len__(self) -> int:
        """int: Number of points in the current object."""
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        """str: Return a string that describes the object."""
        return self.__class__.__name__ + "(\n    " + str(self.tensor) + ")"

    @classmethod
    def cat(cls, points_list: Sequence["BasePoints"]) -> "BasePoints":
        """Concatenate a list of Points into a single Points.

        Args:
            points_list (Sequence[:obj:`BasePoints`]): List of points.

        Returns:
            :obj:`BasePoints`: The concatenated points.
        """
        assert isinstance(points_list, (list, tuple))
        if len(points_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(points, cls) for points in points_list)

        # use torch.cat (v.s. layers.cat)
        # so the returned points never share storage with input
        cat_points = cls(
            torch.cat([p.tensor for p in points_list], dim=0),
            points_dim=points_list[0].points_dim,
            attribute_dims=points_list[0].attribute_dims,
        )
        return cat_points

    def numpy(self) -> np.ndarray:
        """Reload ``numpy`` from self.tensor."""
        return self.tensor.numpy()

    def to(self, device: Union[str, torch.device], *args, **kwargs) -> "BasePoints":
        """Convert current points to a specific device.

        Args:
            device (str or :obj:`torch.device`): The name of the device.

        Returns:
            :obj:`BasePoints`: A new points object on the specific device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.to(device, *args, **kwargs), points_dim=self.points_dim, attribute_dims=self.attribute_dims
        )

    def cpu(self) -> "BasePoints":
        """Convert current points to cpu device.

        Returns:
            :obj:`BasePoints`: A new points object on the cpu device.
        """
        original_type = type(self)
        return original_type(self.tensor.cpu(), points_dim=self.points_dim, attribute_dims=self.attribute_dims)

    def cuda(self, *args, **kwargs) -> "BasePoints":
        """Convert current points to cuda device.

        Returns:
            :obj:`BasePoints`: A new points object on the cuda device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.cuda(*args, **kwargs), points_dim=self.points_dim, attribute_dims=self.attribute_dims
        )

    def clone(self) -> "BasePoints":
        """Clone the points.

        Returns:
            :obj:`BasePoints`: Point object with the same properties as self.
        """
        original_type = type(self)
        return original_type(self.tensor.clone(), points_dim=self.points_dim, attribute_dims=self.attribute_dims)

    def detach(self) -> "BasePoints":
        """Detach the points.

        Returns:
            :obj:`BasePoints`: Point object with the same properties as self.
        """
        original_type = type(self)
        return original_type(self.tensor.detach(), points_dim=self.points_dim, attribute_dims=self.attribute_dims)

    @property
    def device(self) -> torch.device:
        """torch.device: The device of the points are on."""
        return self.tensor.device

    def __iter__(self) -> Iterator[Tensor]:
        """Yield a point as a Tensor at a time.

        Returns:
            Iterator[Tensor]: A point of shape (points_dim, ).
        """
        yield from self.tensor

    def new_point(self, data: Union[Tensor, np.ndarray, Sequence[Sequence[float]]]) -> "BasePoints":
        """Create a new point object with data.

        The new point and its tensor has the similar properties as self and
        self.tensor, respectively.

        Args:
            data (Tensor or np.ndarray or Sequence[Sequence[float]]): Data to
                be copied.

        Returns:
            :obj:`BasePoints`: A new point object with ``data``, the object's
            other properties are similar to ``self``.
        """
        new_tensor = self.tensor.new_tensor(data) if not isinstance(data, Tensor) else data.to(self.device)
        original_type = type(self)
        return original_type(new_tensor, points_dim=self.points_dim, attribute_dims=self.attribute_dims)


class LiDARPoints(BasePoints):
    """Points of instances in LIDAR coordinates.

    Args:
        tensor (Tensor or np.ndarray or Sequence[Sequence[float]]): The points
            data with shape (N, points_dim).
        points_dim (int): Integer indicating the dimension of a point. Each row
            is (x, y, z, ...). Defaults to 3.
        attribute_dims (dict, optional): Dictionary to indicate the meaning of
            extra dimension. Defaults to None.

    Attributes:
        tensor (Tensor): Float matrix with shape (N, points_dim).
        points_dim (int): Integer indicating the dimension of a point. Each row
            is (x, y, z, ...).
        attribute_dims (dict, optional): Dictionary to indicate the meaning of
            extra dimension. Defaults to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(
        self,
        tensor: Union[Tensor, np.ndarray, Sequence[Sequence[float]]],
        points_dim: int = 3,
        attribute_dims: Optional[dict] = None,
    ) -> None:
        super(LiDARPoints, self).__init__(tensor, points_dim=points_dim, attribute_dims=attribute_dims)
        self.rotation_axis = 2

    def flip(self, bev_direction: str = "horizontal") -> None:
        """Flip the points along given BEV direction.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
                Defaults to 'horizontal'.
        """
        assert bev_direction in ("horizontal", "vertical")
        if bev_direction == "horizontal":
            self.tensor[:, 1] = -self.tensor[:, 1]
        elif bev_direction == "vertical":
            self.tensor[:, 0] = -self.tensor[:, 0]

    def convert_to(self, dst: int, rt_mat: Optional[Union[Tensor, np.ndarray]] = None) -> "BasePoints":
        """Convert self to ``dst`` mode.

        Args:
            dst (int): The target Point mode.
            rt_mat (Tensor or np.ndarray, optional): The rotation and
                translation matrix between different coordinates.
                Defaults to None. The conversion from ``src`` coordinates to
                ``dst`` coordinates usually comes along the change of sensors,
                e.g., from camera to LiDAR. This requires a transformation
                matrix.

        Returns:
            :obj:`BasePoints`: The converted point of the same type in the
            ``dst`` mode.
        """
        from mmdet3d.structures.bbox_3d import Coord3DMode

        return Coord3DMode.convert_point(point=self, src=Coord3DMode.LIDAR, dst=dst, rt_mat=rt_mat)


def get_points_type(points_type: str) -> type:
    """Get the class of points according to coordinate type.

    Args:
        points_type (str): The type of points coordinate. The valid value are
            "CAMERA", "LIDAR" and "DEPTH".

    Returns:
        type: Points type.
    """
    points_type_upper = points_type.upper()
    if points_type_upper == "CAMERA":
        points_cls = CameraPoints
    elif points_type_upper == "LIDAR":
        points_cls = LiDARPoints
    elif points_type_upper == "DEPTH":
        points_cls = DepthPoints
    else:
        raise ValueError('Only "points_type" of "CAMERA", "LIDAR" and "DEPTH" ' f"are supported, got {points_type}")

    return points_cls


def get_box_type(box_type: str) -> Tuple[type, int]:
    """Get the type and mode of box structure.

    Args:
        box_type (str): The type of box structure. The valid value are "LiDAR",
            "Camera" and "Depth".

    Raises:
        ValueError: A ValueError is raised when ``box_type`` does not belong to
            the three valid types.

    Returns:
        tuple: Box type and box mode.
    """
    from models.experimental.functional_pointpillars.reference.point_pillars_utils import LiDARInstance3DBoxes

    box_type_lower = box_type.lower()
    if box_type_lower == "lidar":
        box_type_3d = LiDARInstance3DBoxes
        box_mode_3d = Box3DMode.LIDAR
    elif box_type_lower == "camera":
        box_type_3d = CameraInstance3DBoxes
        box_mode_3d = Box3DMode.CAM
    elif box_type_lower == "depth":
        box_type_3d = DepthInstance3DBoxes
        box_mode_3d = Box3DMode.DEPTH
    else:
        raise ValueError('Only "box_type" of "camera", "lidar", "depth" are ' f"supported, got {box_type}")

    return box_type_3d, box_mode_3d


class LoadPointsFromFile(BaseTransform):
    """Load Points From File.

    Required Keys:

    - lidar_points (dict)

        - lidar_path (str)

    Added Keys:

    - points (np.float32)

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:

            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points. Defaults to 6.
        use_dim (list[int] | int): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        norm_intensity (bool): Whether to normlize the intensity. Defaults to
            False.
        norm_elongation (bool): Whether to normlize the elongation. This is
            usually used in Waymo dataset.Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(
        self,
        coord_type: str,
        load_dim: int = 6,
        use_dim: Union[int, List[int]] = [0, 1, 2],
        shift_height: bool = False,
        use_color: bool = False,
        norm_intensity: bool = False,
        norm_elongation: bool = False,
        backend_args: Optional[dict] = None,
    ) -> None:
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, f"Expect all used dimensions < {load_dim}, got {use_dim}"
        print("coord_type", coord_type)
        assert coord_type in ["CAMERA", "LIDAR", "DEPTH"]

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.norm_intensity = norm_intensity
        self.norm_elongation = norm_elongation
        self.backend_args = backend_args

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        try:
            pts_bytes = get(pts_filename, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith(".npy"):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_file_path = results["lidar_points"]["lidar_path"]
        points = self._load_points(pts_file_path)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        if self.norm_intensity:
            assert (
                len(self.use_dim) >= 4
            ), f"When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}"  # noqa: E501
            points[:, 3] = np.tanh(points[:, 3])
        if self.norm_elongation:
            assert (
                len(self.use_dim) >= 5
            ), f"When using elongation norm, expect used dimensions >= 5, got {len(self.use_dim)}"  # noqa: E501
            points[:, 4] = np.tanh(points[:, 4])
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate([points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points = points_class(points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results["points"] = points

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + "("
        repr_str += f"shift_height={self.shift_height}, "
        repr_str += f"use_color={self.use_color}, "
        repr_str += f"backend_args={self.backend_args}, "
        repr_str += f"load_dim={self.load_dim}, "
        repr_str += f"use_dim={self.use_dim})"
        repr_str += f"norm_intensity={self.norm_intensity})"
        repr_str += f"norm_elongation={self.norm_elongation})"
        return repr_str


class LoadPointsFromDict(LoadPointsFromFile):
    """Load Points From Dict."""

    def transform(self, results: dict) -> dict:
        """Convert the type of points from ndarray to corresponding
        `point_class`.

        Args:
            results (dict): input result. The value of key `points` is a
                numpy array.

        Returns:
            dict: The processed results.
        """
        assert "points" in results
        points = results["points"]

        if self.norm_intensity:
            assert (
                len(self.use_dim) >= 4
            ), f"When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}"  # noqa: E501
            points[:, 3] = np.tanh(points[:, 3])
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate([points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(
                    color=[
                        points.shape[1] - 3,
                        points.shape[1] - 2,
                        points.shape[1] - 1,
                    ]
                )
            )

        points_class = get_points_type(self.coord_type)
        points = points_class(points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results["points"] = points
        return results


class LidarDet3DInferencerLoader(BaseTransform):
    """Load point cloud in the Inferencer's pipeline.

    Added keys:
      - points
      - timestamp
      - axis_align_matrix
      - box_type_3d
      - box_mode_3d
    """

    def __init__(self, coord_type="LIDAR", **kwargs) -> None:
        super().__init__()
        self.from_file = LoadPointsFromFile(coord_type=coord_type, **kwargs)
        self.from_ndarray = LoadPointsFromDict(coord_type=coord_type, **kwargs)
        self.box_type_3d, self.box_mode_3d = get_box_type(coord_type)

    def transform(self, single_input: dict) -> dict:
        """Transform function to add image meta information.
        Args:
            single_input (dict): Single input.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        assert "points" in single_input, "key 'points' must be in input dict"
        if isinstance(single_input["points"], str):
            inputs = dict(
                lidar_points=dict(lidar_path=single_input["points"]),
                timestamp=1,
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d,
            )
        elif isinstance(single_input["points"], np.ndarray):
            inputs = dict(
                points=single_input["points"],
                timestamp=1,
                # for ScanNet demo we need axis_align_matrix
                axis_align_matrix=np.eye(4),
                box_type_3d=self.box_type_3d,
                box_mode_3d=self.box_mode_3d,
            )
        else:
            raise ValueError("Unsupported input points type: " f"{type(single_input['points'])}")

        if "points" in inputs:
            return self.from_ndarray(inputs)
        return self.from_file(inputs)


# taken form mmdetection3d/mmdet3d/datasets/transformers/loading.py file
class LoadPointsFromMultiSweeps(BaseTransform):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points. Defaults to False.
        test_mode (bool): If `test_mode=True`, it will not randomly sample
            sweeps but select the nearest N frames. Defaults to False.
    """

    def __init__(
        self,
        sweeps_num: int = 10,
        load_dim: int = 5,
        use_dim: List[int] = [0, 1, 2, 4],
        backend_args: Optional[dict] = None,
        pad_empty_sweeps: bool = False,
        remove_close: bool = False,
        test_mode: bool = False,
    ) -> None:
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, f"Expect all used dimensions < {load_dim}, got {use_dim}"
        self.use_dim = use_dim
        self.backend_args = backend_args
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        try:
            pts_bytes = get(pts_filename, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith(".npy"):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(
        self, points: Union[np.ndarray, BasePoints], radius: float = 1.0
    ) -> Union[np.ndarray, BasePoints]:
        """Remove point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray | :obj:`BasePoints`: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def transform(self, results: dict) -> dict:
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data.
            Updated key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point
                  cloud arrays.
        """
        points = results["points"]
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results["timestamp"]
        if "lidar_sweeps" not in results:
            if self.pad_empty_sweeps:
                for i in range(self.sweeps_num):
                    if self.remove_close:
                        sweep_points_list.append(self._remove_close(points))
                    else:
                        sweep_points_list.append(points)
        else:
            if len(results["lidar_sweeps"]) <= self.sweeps_num:
                choices = np.arange(len(results["lidar_sweeps"]))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(len(results["lidar_sweeps"]), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results["lidar_sweeps"][idx]
                points_sweep = self._load_points(sweep["lidar_points"]["lidar_path"])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                # bc-breaking: Timestamp has divided 1e6 in pkl infos.
                sweep_ts = sweep["timestamp"]
                lidar2sensor = np.array(sweep["lidar_points"]["lidar2sensor"])
                points_sweep[:, :3] = points_sweep[:, :3] @ lidar2sensor[:3, :3]
                points_sweep[:, :3] -= lidar2sensor[:3, 3]
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results["points"] = points
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        return f"{self.__class__.__name__}(sweeps_num={self.sweeps_num})"


# code taken from mmdetection3d/mmdet3d/datasets/transforms/test_time_aug.py
class MultiScaleFlipAug3D(BaseTransform):
    """Test-time augmentation with multiple scales and flipping.

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple]): Images scales for resizing.
        pts_scale_ratio (float | list[float]): Points scale ratios for
            resizing.
        flip (bool): Whether apply flip augmentation. Defaults to False.
        flip_direction (str | list[str]): Flip augmentation directions
            for images, options are "horizontal" and "vertical".
            If flip_direction is list, multiple flip augmentations will
            be applied. It has no effect when ``flip == False``.
            Defaults to 'horizontal'.
        pcd_horizontal_flip (bool): Whether to apply horizontal flip
            augmentation to point cloud. Defaults to False.
            Note that it works only when 'flip' is turned on.
        pcd_vertical_flip (bool): Whether to apply vertical flip
            augmentation to point cloud. Defaults to False.
            Note that it works only when 'flip' is turned on.
    """

    def __init__(
        self,
        transforms: List[dict],
        img_scale: Optional[Union[Tuple[int], List[Tuple[int]]]],
        pts_scale_ratio: Union[float, List[float]],
        flip: bool = False,
        flip_direction: str = "horizontal",
        pcd_horizontal_flip: bool = False,
        pcd_vertical_flip: bool = False,
    ) -> None:
        self.transforms = Compose(transforms)
        self.img_scale = img_scale if isinstance(img_scale, list) else [img_scale]
        self.pts_scale_ratio = pts_scale_ratio if isinstance(pts_scale_ratio, list) else [float(pts_scale_ratio)]

        assert mmengine.is_list_of(self.img_scale, tuple)
        assert mmengine.is_list_of(self.pts_scale_ratio, float)

        self.flip = flip
        self.pcd_horizontal_flip = pcd_horizontal_flip
        self.pcd_vertical_flip = pcd_vertical_flip

        self.flip_direction = flip_direction if isinstance(flip_direction, list) else [flip_direction]
        assert mmengine.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ["horizontal"]:
            warnings.warn("flip_direction has no effect when flip is set to False")
        if self.flip and not any([(t["type"] == "RandomFlip3D" or t["type"] == "RandomFlip") for t in transforms]):
            warnings.warn("flip has no effect when RandomFlip is not in transforms")

    def transform(self, results: Dict) -> List[Dict]:
        """Call function to augment common fields in results.

        Args:
            results (dict): Result dict contains the data to augment.

        Returns:
            List[dict]: The list contains the data that is augmented with
            different scales and flips.
        """
        aug_data_list = []

        # modified from `flip_aug = [False, True] if self.flip else [False]`
        # to reduce unnecessary scenes when using double flip augmentation
        # during test time
        flip_aug = [True] if self.flip else [False]
        pcd_horizontal_flip_aug = [False, True] if self.flip and self.pcd_horizontal_flip else [False]
        pcd_vertical_flip_aug = [False, True] if self.flip and self.pcd_vertical_flip else [False]
        for scale in self.img_scale:
            # TODO refactor according to augtest docs
            self.transforms.transforms[0].scale = scale
            for pts_scale_ratio in self.pts_scale_ratio:
                for flip in flip_aug:
                    for pcd_horizontal_flip in pcd_horizontal_flip_aug:
                        for pcd_vertical_flip in pcd_vertical_flip_aug:
                            for direction in self.flip_direction:
                                # results.copy will cause bug
                                # since it is shallow copy
                                _results = deepcopy(results)
                                _results["scale"] = scale
                                _results["flip"] = flip
                                _results["pcd_scale_factor"] = pts_scale_ratio
                                _results["flip_direction"] = direction
                                _results["pcd_horizontal_flip"] = pcd_horizontal_flip
                                _results["pcd_vertical_flip"] = pcd_vertical_flip
                                data = self.transforms(_results)
                                aug_data_list.append(data)

        return aug_data_list

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(transforms={self.transforms}, "
        repr_str += f"img_scale={self.img_scale}, flip={self.flip}, "
        repr_str += f"pts_scale_ratio={self.pts_scale_ratio}, "
        repr_str += f"flip_direction={self.flip_direction})"
        return repr_str


# taken from mmdetection3d/mmdet3d/datasets/transforms/transforms_3d.py
class GlobalRotScaleTrans(BaseTransform):
    """Apply global rotation, scaling and translation to a 3D scene.

    Required Keys:

    - points (np.float32)
    - gt_bboxes_3d (np.float32)

    Modified Keys:

    - points (np.float32)
    - gt_bboxes_3d (np.float32)

    Added Keys:

    - points (np.float32)
    - pcd_trans (np.float32)
    - pcd_rotation (np.float32)
    - pcd_rotation_angle (np.float32)
    - pcd_scale_factor (np.float32)

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float]): The standard deviation of
            translation noise applied to a scene, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0].
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(
        self,
        rot_range: List[float] = [-0.78539816, 0.78539816],
        scale_ratio_range: List[float] = [0.95, 1.05],
        translation_std: List[int] = [0, 0, 0],
        shift_height: bool = False,
    ) -> None:
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), f"unsupported rot_range type {type(rot_range)}"
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), f"unsupported scale_ratio_range type {type(scale_ratio_range)}"

        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(
                translation_std, (int, float)
            ), f"unsupported translation_std type {type(translation_std)}"
            translation_std = [translation_std, translation_std, translation_std]
        assert all([std >= 0 for std in translation_std]), "translation_std should be positive"
        self.translation_std = translation_std
        self.shift_height = shift_height

    def _trans_bbox_points(self, input_dict: dict) -> None:
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans'
            and `gt_bboxes_3d` is updated in the result dict.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T

        input_dict["points"].translate(trans_factor)
        input_dict["pcd_trans"] = trans_factor
        if "gt_bboxes_3d" in input_dict:
            input_dict["gt_bboxes_3d"].translate(trans_factor)

    def _rot_bbox_points(self, input_dict: dict) -> None:
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation'
            and `gt_bboxes_3d` is updated in the result dict.
        """
        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        if "gt_bboxes_3d" in input_dict and len(input_dict["gt_bboxes_3d"].tensor) != 0:
            # rotate points with bboxes
            points, rot_mat_T = input_dict["gt_bboxes_3d"].rotate(noise_rotation, input_dict["points"])
            input_dict["points"] = points
        else:
            # if no bbox in input_dict, only rotate points
            rot_mat_T = input_dict["points"].rotate(noise_rotation)

        input_dict["pcd_rotation"] = rot_mat_T
        input_dict["pcd_rotation_angle"] = noise_rotation

    def _scale_bbox_points(self, input_dict: dict) -> None:
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points' and
            `gt_bboxes_3d` is updated in the result dict.
        """
        scale = input_dict["pcd_scale_factor"]
        points = input_dict["points"]
        points.scale(scale)
        if self.shift_height:
            assert (
                "height" in points.attribute_dims.keys()
            ), "setting shift_height=True but points have no height attribute"
            points.tensor[:, points.attribute_dims["height"]] *= scale
        input_dict["points"] = points

        if "gt_bboxes_3d" in input_dict and len(input_dict["gt_bboxes_3d"].tensor) != 0:
            input_dict["gt_bboxes_3d"].scale(scale)

    def _random_scale(self, input_dict: dict) -> None:
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor'
            are updated in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0], self.scale_ratio_range[1])
        input_dict["pcd_scale_factor"] = scale_factor

    def transform(self, input_dict: dict) -> dict:
        """Private function to rotate, scale and translate bounding boxes and
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans' and `gt_bboxes_3d` are updated
            in the result dict.
        """
        if "transformation_3d_flow" not in input_dict:
            input_dict["transformation_3d_flow"] = []

        self._rot_bbox_points(input_dict)

        if "pcd_scale_factor" not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)

        input_dict["transformation_3d_flow"].extend(["R", "S", "T"])
        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(rot_range={self.rot_range},"
        repr_str += f" scale_ratio_range={self.scale_ratio_range},"
        repr_str += f" translation_std={self.translation_std},"
        repr_str += f" shift_height={self.shift_height})"
        return repr_str


def autocast_box_type(dst_box_type="hbox") -> Callable:
    """A decorator which automatically casts results['gt_bboxes'] to the
    destination box type.

    It commenly used in mmdet.datasets.transforms to make the transforms up-
    compatible with the np.ndarray type of results['gt_bboxes'].

    The speed of processing of np.ndarray and BaseBoxes data are the same:

    - np.ndarray: 0.0509 img/s
    - BaseBoxes: 0.0551 img/s

    Args:
        dst_box_type (str): Destination box type.
    """
    _, box_type_cls = get_box_type(dst_box_type)

    def decorator(func: Callable) -> Callable:
        def wrapper(self, results: dict, *args, **kwargs) -> dict:
            if "gt_bboxes" not in results or isinstance(results["gt_bboxes"], BaseBoxes):
                return func(self, results)
            elif isinstance(results["gt_bboxes"], np.ndarray):
                results["gt_bboxes"] = box_type_cls(results["gt_bboxes"], clone=False)
                if "mix_results" in results:
                    for res in results["mix_results"]:
                        if isinstance(res["gt_bboxes"], np.ndarray):
                            res["gt_bboxes"] = box_type_cls(res["gt_bboxes"], clone=False)

                _results = func(self, results, *args, **kwargs)

                # In some cases, the function will process gt_bboxes in-place
                # Simultaneously convert inputting and outputting gt_bboxes
                # back to np.ndarray
                if isinstance(_results, dict) and "gt_bboxes" in _results:
                    if isinstance(_results["gt_bboxes"], BaseBoxes):
                        _results["gt_bboxes"] = _results["gt_bboxes"].numpy()
                if isinstance(results["gt_bboxes"], BaseBoxes):
                    results["gt_bboxes"] = results["gt_bboxes"].numpy()
                return _results
            else:
                raise TypeError(
                    "auto_box_type requires results['gt_bboxes'] to "
                    "be BaseBoxes or np.ndarray, but got "
                    f"{type(results['gt_bboxes'])}"
                )

        return wrapper

    return decorator


from mmcv.transforms import RandomFlip as MMCV_RandomFlip


class RandomFlip(MMCV_RandomFlip):  # removed by me to remove the dependency
    """Flip the image & bbox & mask & segmentation map. Added or Updated keys:
    flip, flip_direction, img, gt_bboxes, and gt_seg_map. There are 3 flip
    modes:

     - ``prob`` is float, ``direction`` is string: the image will be
         ``direction``ly flipped with probability of ``prob`` .
         E.g., ``prob=0.5``, ``direction='horizontal'``,
         then image will be horizontally flipped with probability of 0.5.
     - ``prob`` is float, ``direction`` is list of string: the image will
         be ``direction[i]``ly flipped with probability of
         ``prob/len(direction)``.
         E.g., ``prob=0.5``, ``direction=['horizontal', 'vertical']``,
         then image will be horizontally flipped with probability of 0.25,
         vertically with probability of 0.25.
     - ``prob`` is list of float, ``direction`` is list of string:
         given ``len(prob) == len(direction)``, the image will
         be ``direction[i]``ly flipped with probability of ``prob[i]``.
         E.g., ``prob=[0.3, 0.5]``, ``direction=['horizontal',
         'vertical']``, then image will be horizontally flipped with
         probability of 0.3, vertically with probability of 0.5.


    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_masks
    - gt_seg_map

    Added Keys:

    - flip
    - flip_direction
    - homography_matrix


    Args:
         prob (float | list[float], optional): The flipping probability.
             Defaults to None.
         direction(str | list[str]): The flipping direction. Options
             If input is a list, the length must equal ``prob``. Each
             element in ``prob`` indicates the flip probability of
             corresponding direction. Defaults to 'horizontal'.
    """

    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the RandomFlip."""
        cur_dir = results["flip_direction"]
        h, w = results["img"].shape[:2]

        if cur_dir == "horizontal":
            homography_matrix = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        elif cur_dir == "vertical":
            homography_matrix = np.array([[1, 0, 0], [0, -1, h], [0, 0, 1]], dtype=np.float32)
        elif cur_dir == "diagonal":
            homography_matrix = np.array([[-1, 0, w], [0, -1, h], [0, 0, 1]], dtype=np.float32)
        else:
            homography_matrix = np.eye(3, dtype=np.float32)

        if results.get("homography_matrix", None) is None:
            results["homography_matrix"] = homography_matrix
        else:
            results["homography_matrix"] = homography_matrix @ results["homography_matrix"]

    # @autocast_box_type()#addedbyme
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, and semantic segmentation map."""
        # flip image
        results["img"] = mmcv.imflip(results["img"], direction=results["flip_direction"])

        img_shape = results["img"].shape[:2]

        # flip bboxes
        if results.get("gt_bboxes", None) is not None:
            results["gt_bboxes"].flip_(img_shape, results["flip_direction"])

        # flip masks
        if results.get("gt_masks", None) is not None:
            results["gt_masks"] = results["gt_masks"].flip(results["flip_direction"])

        # flip segs
        if results.get("gt_seg_map", None) is not None:
            results["gt_seg_map"] = mmcv.imflip(results["gt_seg_map"], direction=results["flip_direction"])

        # record homography matrix for flip
        self._record_homography_matrix(results)


class RandomFlip3D(RandomFlip):
    """Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Required Keys:

    - points (np.float32)
    - gt_bboxes_3d (np.float32)

    Modified Keys:

    - points (np.float32)
    - gt_bboxes_3d (np.float32)

    Added Keys:

    - points (np.float32)
    - pcd_trans (np.float32)
    - pcd_rotation (np.float32)
    - pcd_rotation_angle (np.float32)
    - pcd_scale_factor (np.float32)

    Args:
        sync_2d (bool): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float): The flipping probability
            in vertical direction. Defaults to 0.0.
        flip_box3d (bool): Whether to flip bounding box. In most of the case,
            the box should be fliped. In cam-based bev detection, this is set
            to False, since the flip of 2D images does not influence the 3D
            box. Defaults to True.
    """

    def __init__(
        self,
        sync_2d: bool = True,
        flip_ratio_bev_horizontal: float = 0.0,
        flip_ratio_bev_vertical: float = 0.0,
        flip_box3d: bool = True,
        **kwargs,
    ) -> None:
        # `flip_ratio_bev_horizontal` is equal to
        # for flip prob of 2d image when
        # `sync_2d` is True
        super(RandomFlip3D, self).__init__(prob=flip_ratio_bev_horizontal, direction="horizontal", **kwargs)
        self.sync_2d = sync_2d
        self.flip_ratio_bev_horizontal = flip_ratio_bev_horizontal
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        self.flip_box3d = flip_box3d
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(flip_ratio_bev_horizontal, (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(flip_ratio_bev_vertical, (int, float)) and 0 <= flip_ratio_bev_vertical <= 1

    def random_flip_data_3d(self, input_dict: dict, direction: str = "horizontal") -> None:
        """Flip 3D data randomly.

        `random_flip_data_3d` should take these situations into consideration:

        - 1. LIDAR-based 3d detection
        - 2. LIDAR-based 3d segmentation
        - 3. vision-only detection
        - 4. multi-modality 3d detection.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Defaults to 'horizontal'.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are
            updated in the result dict.
        """
        assert direction in ["horizontal", "vertical"]
        if self.flip_box3d:
            if "gt_bboxes_3d" in input_dict:
                if "points" in input_dict:
                    input_dict["points"] = input_dict["gt_bboxes_3d"].flip(direction, points=input_dict["points"])
                else:
                    # vision-only detection
                    input_dict["gt_bboxes_3d"].flip(direction)
            else:
                input_dict["points"].flip(direction)

        if "centers_2d" in input_dict:
            assert (
                self.sync_2d is True and direction == "horizontal"
            ), "Only support sync_2d=True and horizontal flip with images"
            w = input_dict["img_shape"][1]
            input_dict["centers_2d"][..., 0] = w - input_dict["centers_2d"][..., 0]
            # need to modify the horizontal position of camera center
            # along u-axis in the image (flip like centers2d)
            # ['cam2img'][0][2] = c_u
            # see more details and examples at
            # https://github.com/open-mmlab/mmdetection3d/pull/744
            input_dict["cam2img"][0][2] = w - input_dict["cam2img"][0][2]

    def _flip_on_direction(self, results: dict) -> None:
        """Function to flip images, bounding boxes, semantic segmentation map
        and keypoints.

        Add the override feature that if 'flip' is already in results, use it
        to do the augmentation.
        """
        if "flip" not in results:
            cur_dir = self._choose_direction()
        else:
            # `flip_direction` works only when `flip` is True.
            # For example, in `MultiScaleFlipAug3D`, `flip_direction` is
            # 'horizontal' but `flip` is False.
            if results["flip"]:
                assert "flip_direction" in results, "flip and flip_direction "
                "must exist simultaneously"
                cur_dir = results["flip_direction"]
            else:
                cur_dir = None
        if cur_dir is None:
            results["flip"] = False
            results["flip_direction"] = None
        else:
            results["flip"] = True
            results["flip_direction"] = cur_dir
            self._flip(results)

    def transform(self, input_dict: dict) -> dict:
        """Call function to flip points, values in the ``bbox3d_fields`` and
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction',
            'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added
            into result dict.
        """
        # flip 2D image and its annotations
        if "img" in input_dict:
            super(RandomFlip3D, self).transform(input_dict)

        if self.sync_2d and "img" in input_dict:
            input_dict["pcd_horizontal_flip"] = input_dict["flip"]
            input_dict["pcd_vertical_flip"] = False
        else:
            if "pcd_horizontal_flip" not in input_dict:
                flip_horizontal = True if np.random.rand() < self.flip_ratio_bev_horizontal else False
                input_dict["pcd_horizontal_flip"] = flip_horizontal
            if "pcd_vertical_flip" not in input_dict:
                flip_vertical = True if np.random.rand() < self.flip_ratio_bev_vertical else False
                input_dict["pcd_vertical_flip"] = flip_vertical

        if "transformation_3d_flow" not in input_dict:
            input_dict["transformation_3d_flow"] = []

        if input_dict["pcd_horizontal_flip"]:
            self.random_flip_data_3d(input_dict, "horizontal")
            input_dict["transformation_3d_flow"].extend(["HF"])
        if input_dict["pcd_vertical_flip"]:
            self.random_flip_data_3d(input_dict, "vertical")
            input_dict["transformation_3d_flow"].extend(["VF"])
        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(sync_2d={self.sync_2d},"
        repr_str += f" flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})"
        return repr_str


# taken from mmdet3d/datasets/transforms/transforms_3d.py
class PointsRangeFilter(BaseTransform):
    """Filter points by the range.

    Required Keys:

    - points
    - pts_instance_mask (optional)

    Modified Keys:

    - points
    - pts_instance_mask (optional)

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range: List[float]) -> None:
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def transform(self, input_dict: dict) -> dict:
        """Transform function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask'
            and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = input_dict["points"]
        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = points[points_mask]
        input_dict["points"] = clean_points
        points_mask = points_mask.numpy()

        pts_instance_mask = input_dict.get("pts_instance_mask", None)
        pts_semantic_mask = input_dict.get("pts_semantic_mask", None)

        if pts_instance_mask is not None:
            input_dict["pts_instance_mask"] = pts_instance_mask[points_mask]

        if pts_semantic_mask is not None:
            input_dict["pts_semantic_mask"] = pts_semantic_mask[points_mask]

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(point_cloud_range={self.pcd_range.tolist()})"
        return repr_str


from mmengine.structures import InstanceData
from mmengine.structures import BaseDataElement, InstanceData, PixelData


class DetDataSample(BaseDataElement):
    """A data structure interface of MMDetection. They are used as interfaces
    between different components.

    The attributes in ``DetDataSample`` are divided into several parts:

        - ``proposals``(InstanceData): Region proposals used in two-stage
            detectors.
        - ``gt_instances``(InstanceData): Ground truth of instance annotations.
        - ``pred_instances``(InstanceData): Instances of detection predictions.
        - ``pred_track_instances``(InstanceData): Instances of tracking
            predictions.
        - ``ignored_instances``(InstanceData): Instances to be ignored during
            training/testing.
        - ``gt_panoptic_seg``(PixelData): Ground truth of panoptic
            segmentation.
        - ``pred_panoptic_seg``(PixelData): Prediction of panoptic
           segmentation.
        - ``gt_sem_seg``(PixelData): Ground truth of semantic segmentation.
        - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.

    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine.structures import InstanceData
         >>> from mmdet.structures import DetDataSample

         >>> data_sample = DetDataSample()
         >>> img_meta = dict(img_shape=(800, 1196),
         ...                 pad_shape=(800, 1216))
         >>> gt_instances = InstanceData(metainfo=img_meta)
         >>> gt_instances.bboxes = torch.rand((5, 4))
         >>> gt_instances.labels = torch.rand((5,))
         >>> data_sample.gt_instances = gt_instances
         >>> assert 'img_shape' in data_sample.gt_instances.metainfo_keys()
         >>> len(data_sample.gt_instances)
         5
         >>> print(data_sample)
        <DetDataSample(

            META INFORMATION

            DATA FIELDS
            gt_instances: <InstanceData(

                    META INFORMATION
                    pad_shape: (800, 1216)
                    img_shape: (800, 1196)

                    DATA FIELDS
                    labels: tensor([0.8533, 0.1550, 0.5433, 0.7294, 0.5098])
                    bboxes:
                    tensor([[9.7725e-01, 5.8417e-01, 1.7269e-01, 6.5694e-01],
                            [1.7894e-01, 5.1780e-01, 7.0590e-01, 4.8589e-01],
                            [7.0392e-01, 6.6770e-01, 1.7520e-01, 1.4267e-01],
                            [2.2411e-01, 5.1962e-01, 9.6953e-01, 6.6994e-01],
                            [4.1338e-01, 2.1165e-01, 2.7239e-04, 6.8477e-01]])
                ) at 0x7f21fb1b9190>
        ) at 0x7f21fb1b9880>
         >>> pred_instances = InstanceData(metainfo=img_meta)
         >>> pred_instances.bboxes = torch.rand((5, 4))
         >>> pred_instances.scores = torch.rand((5,))
         >>> data_sample = DetDataSample(pred_instances=pred_instances)
         >>> assert 'pred_instances' in data_sample

         >>> pred_track_instances = InstanceData(metainfo=img_meta)
         >>> pred_track_instances.bboxes = torch.rand((5, 4))
         >>> pred_track_instances.scores = torch.rand((5,))
         >>> data_sample = DetDataSample(
         ...    pred_track_instances=pred_track_instances)
         >>> assert 'pred_track_instances' in data_sample

         >>> data_sample = DetDataSample()
         >>> gt_instances_data = dict(
         ...                        bboxes=torch.rand(2, 4),
         ...                        labels=torch.rand(2),
         ...                        masks=np.random.rand(2, 2, 2))
         >>> gt_instances = InstanceData(**gt_instances_data)
         >>> data_sample.gt_instances = gt_instances
         >>> assert 'gt_instances' in data_sample
         >>> assert 'masks' in data_sample.gt_instances

         >>> data_sample = DetDataSample()
         >>> gt_panoptic_seg_data = dict(panoptic_seg=torch.rand(2, 4))
         >>> gt_panoptic_seg = PixelData(**gt_panoptic_seg_data)
         >>> data_sample.gt_panoptic_seg = gt_panoptic_seg
         >>> print(data_sample)
        <DetDataSample(

            META INFORMATION

            DATA FIELDS
            _gt_panoptic_seg: <BaseDataElement(

                    META INFORMATION

                    DATA FIELDS
                    panoptic_seg: tensor([[0.7586, 0.1262, 0.2892, 0.9341],
                                [0.3200, 0.7448, 0.1052, 0.5371]])
                ) at 0x7f66c2bb7730>
            gt_panoptic_seg: <BaseDataElement(

                    META INFORMATION

                    DATA FIELDS
                    panoptic_seg: tensor([[0.7586, 0.1262, 0.2892, 0.9341],
                                [0.3200, 0.7448, 0.1052, 0.5371]])
                ) at 0x7f66c2bb7730>
        ) at 0x7f66c2bb7280>
        >>> data_sample = DetDataSample()
        >>> gt_segm_seg_data = dict(segm_seg=torch.rand(2, 2, 2))
        >>> gt_segm_seg = PixelData(**gt_segm_seg_data)
        >>> data_sample.gt_segm_seg = gt_segm_seg
        >>> assert 'gt_segm_seg' in data_sample
        >>> assert 'segm_seg' in data_sample.gt_segm_seg
    """

    @property
    def proposals(self) -> InstanceData:
        return self._proposals

    @proposals.setter
    def proposals(self, value: InstanceData):
        self.set_field(value, "_proposals", dtype=InstanceData)

    @proposals.deleter
    def proposals(self):
        del self._proposals

    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        self.set_field(value, "_gt_instances", dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        self.set_field(value, "_pred_instances", dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances

    # directly add ``pred_track_instances`` in ``DetDataSample``
    # so that the ``TrackDataSample`` does not bother to access the
    # instance-level information.
    @property
    def pred_track_instances(self) -> InstanceData:
        return self._pred_track_instances

    @pred_track_instances.setter
    def pred_track_instances(self, value: InstanceData):
        self.set_field(value, "_pred_track_instances", dtype=InstanceData)

    @pred_track_instances.deleter
    def pred_track_instances(self):
        del self._pred_track_instances

    @property
    def ignored_instances(self) -> InstanceData:
        return self._ignored_instances

    @ignored_instances.setter
    def ignored_instances(self, value: InstanceData):
        self.set_field(value, "_ignored_instances", dtype=InstanceData)

    @ignored_instances.deleter
    def ignored_instances(self):
        del self._ignored_instances

    @property
    def gt_panoptic_seg(self) -> PixelData:
        return self._gt_panoptic_seg

    @gt_panoptic_seg.setter
    def gt_panoptic_seg(self, value: PixelData):
        self.set_field(value, "_gt_panoptic_seg", dtype=PixelData)

    @gt_panoptic_seg.deleter
    def gt_panoptic_seg(self):
        del self._gt_panoptic_seg

    @property
    def pred_panoptic_seg(self) -> PixelData:
        return self._pred_panoptic_seg

    @pred_panoptic_seg.setter
    def pred_panoptic_seg(self, value: PixelData):
        self.set_field(value, "_pred_panoptic_seg", dtype=PixelData)

    @pred_panoptic_seg.deleter
    def pred_panoptic_seg(self):
        del self._pred_panoptic_seg

    @property
    def gt_sem_seg(self) -> PixelData:
        return self._gt_sem_seg

    @gt_sem_seg.setter
    def gt_sem_seg(self, value: PixelData):
        self.set_field(value, "_gt_sem_seg", dtype=PixelData)

    @gt_sem_seg.deleter
    def gt_sem_seg(self):
        del self._gt_sem_seg

    @property
    def pred_sem_seg(self) -> PixelData:
        return self._pred_sem_seg

    @pred_sem_seg.setter
    def pred_sem_seg(self, value: PixelData):
        self.set_field(value, "_pred_sem_seg", dtype=PixelData)

    @pred_sem_seg.deleter
    def pred_sem_seg(self):
        del self._pred_sem_seg


from collections.abc import Sized

IndexType = Union[
    str, slice, int, list, torch.LongTensor, torch.cuda.LongTensor, torch.BoolTensor, torch.cuda.BoolTensor, np.ndarray
]


class PointData(BaseDataElement):
    """Data structure for point-level annotations or predictions.

    All data items in ``data_fields`` of ``PointData`` meet the following
    requirements:

    - They are all one dimension.
    - They should have the same length.

    `PointData` is used to save point-level semantic and instance mask,
    it also can save `instances_labels` and `instances_scores` temporarily.
    In the future, we would consider to move the instance-level info into
    `gt_instances_3d` and `pred_instances_3d`.

    Examples:
        >>> metainfo = dict(
        ...     sample_idx=random.randint(0, 100))
        >>> points = np.random.randint(0, 255, (100, 3))
        >>> point_data = PointData(metainfo=metainfo,
        ...                        points=points)
        >>> print(len(point_data))
        100

        >>> # slice
        >>> slice_data = point_data[10:60]
        >>> assert len(slice_data) == 50

        >>> # set
        >>> point_data.pts_semantic_mask = torch.randint(0, 255, (100,))
        >>> point_data.pts_instance_mask = torch.randint(0, 255, (100,))
        >>> assert tuple(point_data.pts_semantic_mask.shape) == (100,)
        >>> assert tuple(point_data.pts_instance_mask.shape) == (100,)
    """

    def __setattr__(self, name: str, value: Sized) -> None:
        """setattr is only used to set data.

        The value must have the attribute of `__len__` and have the same length
        of `PointData`.
        """
        if name in ("_metainfo_fields", "_data_fields"):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f"{name} has been used as a " "private attribute, which is immutable.")

        else:
            assert isinstance(value, Sized), "value must contain `__len__` attribute"
            # TODO: make sure the input value share the same length
            super().__setattr__(name, value)

    __setitem__ = __setattr__

    def __getitem__(self, item: IndexType) -> "PointData":
        """
        Args:
            item (str, int, list, :obj:`slice`, :obj:`numpy.ndarray`,
                :obj:`torch.LongTensor`, :obj:`torch.BoolTensor`):
                Get the corresponding values according to item.

        Returns:
            :obj:`PointData`: Corresponding values.
        """
        if isinstance(item, list):
            item = np.array(item)
        if isinstance(item, np.ndarray):
            # The default int type of numpy is platform dependent, int32 for
            # windows and int64 for linux. `torch.Tensor` requires the index
            # should be int64, therefore we simply convert it to int64 here.
            # Mode details in https://github.com/numpy/numpy/issues/9464
            item = item.astype(np.int64) if item.dtype == np.int32 else item
            item = torch.from_numpy(item)
        assert isinstance(
            item, (str, slice, int, torch.LongTensor, torch.cuda.LongTensor, torch.BoolTensor, torch.cuda.BoolTensor)
        )

        if isinstance(item, str):
            return getattr(self, item)

        if isinstance(item, int):
            if item >= len(self) or item < -len(self):  # type: ignore
                raise IndexError(f"Index {item} out of range!")
            else:
                # keep the dimension
                item = slice(item, None, len(self))

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, torch.Tensor):
            assert item.dim() == 1, "Only support to get the" " values along the first dimension."
            if isinstance(item, (torch.BoolTensor, torch.cuda.BoolTensor)):
                assert len(item) == len(self), (
                    "The shape of the "
                    "input(BoolTensor) "
                    f"{len(item)} "
                    "does not match the shape "
                    "of the indexed tensor "
                    "in results_field "
                    f"{len(self)} at "
                    "first dimension."
                )

            for k, v in self.items():
                if isinstance(v, torch.Tensor):
                    new_data[k] = v[item]
                elif isinstance(v, np.ndarray):
                    new_data[k] = v[item.cpu().numpy()]
                elif isinstance(v, (str, list, tuple)) or (hasattr(v, "__getitem__") and hasattr(v, "cat")):
                    # convert to indexes from BoolTensor
                    if isinstance(item, (torch.BoolTensor, torch.cuda.BoolTensor)):
                        indexes = torch.nonzero(item).view(-1).cpu().numpy().tolist()
                    else:
                        indexes = item.cpu().numpy().tolist()
                    slice_list = []
                    if indexes:
                        for index in indexes:
                            slice_list.append(slice(index, None, len(v)))
                    else:
                        slice_list.append(slice(None, 0, None))
                    r_list = [v[s] for s in slice_list]
                    if isinstance(v, (str, list, tuple)):
                        new_value = r_list[0]
                        for r in r_list[1:]:
                            new_value = new_value + r
                    else:
                        new_value = v.cat(r_list)
                    new_data[k] = new_value
                else:
                    raise ValueError(
                        f"The type of `{k}` is `{type(v)}`, which has no "
                        "attribute of `cat`, so it does not "
                        "support slice with `bool`"
                    )
        else:
            # item is a slice
            for k, v in self.items():
                new_data[k] = v[item]
        return new_data  # type: ignore

    def __len__(self) -> int:
        """int: The length of `PointData`."""
        if len(self._data_fields) > 0:
            return len(self.values()[0])
        else:
            return 0


class Det3DDataSample(DetDataSample):
    """A data structure interface of MMDetection3D. They are used as interfaces
    between different components.

    The attributes in ``Det3DDataSample`` are divided into several parts:

        - ``proposals`` (InstanceData): Region proposals used in two-stage
          detectors.
        - ``ignored_instances`` (InstanceData): Instances to be ignored during
          training/testing.
        - ``gt_instances_3d`` (InstanceData): Ground truth of 3D instance
          annotations.
        - ``gt_instances`` (InstanceData): Ground truth of 2D instance
          annotations.
        - ``pred_instances_3d`` (InstanceData): 3D instances of model
          predictions.
          - For point-cloud 3D object detection task whose input modality is
            `use_lidar=True, use_camera=False`, the 3D predictions results are
            saved in `pred_instances_3d`.
          - For vision-only (monocular/multi-view) 3D object detection task
            whose input modality is `use_lidar=False, use_camera=True`, the 3D
            predictions are saved in `pred_instances_3d`.
        - ``pred_instances`` (InstanceData): 2D instances of model predictions.
          - For multi-modality 3D detection task whose input modality is
            `use_lidar=True, use_camera=True`, the 2D predictions are saved in
            `pred_instances`.
        - ``pts_pred_instances_3d`` (InstanceData): 3D instances of model
          predictions based on point cloud.
          - For multi-modality 3D detection task whose input modality is
            `use_lidar=True, use_camera=True`, the 3D predictions based on
            point cloud are saved in `pts_pred_instances_3d` to distinguish
            with `img_pred_instances_3d` which based on image.
        - ``img_pred_instances_3d`` (InstanceData): 3D instances of model
          predictions based on image.
          - For multi-modality 3D detection task whose input modality is
            `use_lidar=True, use_camera=True`, the 3D predictions based on
            image are saved in `img_pred_instances_3d` to distinguish with
            `pts_pred_instances_3d` which based on point cloud.
        - ``gt_pts_seg`` (PointData): Ground truth of point cloud segmentation.
        - ``pred_pts_seg`` (PointData): Prediction of point cloud segmentation.
        - ``eval_ann_info`` (dict or None): Raw annotation, which will be
          passed to evaluator and do the online evaluation.

    Examples:
        >>> import torch
        >>> from mmengine.structures import InstanceData

        >>> from mmdet3d.structures import Det3DDataSample
        >>> from mmdet3d.structures.bbox_3d import BaseInstance3DBoxes

        >>> data_sample = Det3DDataSample()
        >>> meta_info = dict(
        ...     img_shape=(800, 1196, 3),
        ...     pad_shape=(800, 1216, 3))
        >>> gt_instances_3d = InstanceData(metainfo=meta_info)
        >>> gt_instances_3d.bboxes_3d = BaseInstance3DBoxes(torch.rand((5, 7)))
        >>> gt_instances_3d.labels_3d = torch.randint(0, 3, (5,))
        >>> data_sample.gt_instances_3d = gt_instances_3d
        >>> assert 'img_shape' in data_sample.gt_instances_3d.metainfo_keys()
        >>> len(data_sample.gt_instances_3d)
        5
        >>> print(data_sample)
        <Det3DDataSample(
            META INFORMATION
            DATA FIELDS
            gt_instances_3d: <InstanceData(
                    META INFORMATION
                    img_shape: (800, 1196, 3)
                    pad_shape: (800, 1216, 3)
                    DATA FIELDS
                    labels_3d: tensor([1, 0, 2, 0, 1])
                    bboxes_3d: BaseInstance3DBoxes(
                            tensor([[1.9115e-01, 3.6061e-01, 6.7707e-01, 5.2902e-01, 8.0736e-01, 8.2759e-01,
                                2.4328e-01],
                                [5.6272e-01, 2.7508e-01, 5.7966e-01, 9.2410e-01, 3.0456e-01, 1.8912e-01,
                                3.3176e-01],
                                [8.1069e-01, 2.8684e-01, 7.7689e-01, 9.2397e-02, 5.5849e-01, 3.8007e-01,
                                4.6719e-01],
                                [6.6346e-01, 4.8005e-01, 5.2318e-02, 4.4137e-01, 4.1163e-01, 8.9339e-01,
                                7.2847e-01],
                                [2.4800e-01, 7.1944e-01, 3.4766e-01, 7.8583e-01, 8.5507e-01, 6.3729e-02,
                                7.5161e-05]]))
                ) at 0x7f7e29de3a00>
        ) at 0x7f7e2a0e8640>
        >>> pred_instances = InstanceData(metainfo=meta_info)
        >>> pred_instances.bboxes = torch.rand((5, 4))
        >>> pred_instances.scores = torch.rand((5, ))
        >>> data_sample = Det3DDataSample(pred_instances=pred_instances)
        >>> assert 'pred_instances' in data_sample

        >>> pred_instances_3d = InstanceData(metainfo=meta_info)
        >>> pred_instances_3d.bboxes_3d = BaseInstance3DBoxes(
        ...     torch.rand((5, 7)))
        >>> pred_instances_3d.scores_3d = torch.rand((5, ))
        >>> pred_instances_3d.labels_3d = torch.rand((5, ))
        >>> data_sample = Det3DDataSample(pred_instances_3d=pred_instances_3d)
        >>> assert 'pred_instances_3d' in data_sample

        >>> data_sample = Det3DDataSample()
        >>> gt_instances_3d_data = dict(
        ...     bboxes_3d=BaseInstance3DBoxes(torch.rand((2, 7))),
        ...     labels_3d=torch.rand(2))
        >>> gt_instances_3d = InstanceData(**gt_instances_3d_data)
        >>> data_sample.gt_instances_3d = gt_instances_3d
        >>> assert 'gt_instances_3d' in data_sample
        >>> assert 'bboxes_3d' in data_sample.gt_instances_3d

        >>> from mmdet3d.structures import PointData
        >>> data_sample = Det3DDataSample()
        >>> gt_pts_seg_data = dict(
        ...     pts_instance_mask=torch.rand(2),
        ...     pts_semantic_mask=torch.rand(2))
        >>> data_sample.gt_pts_seg = PointData(**gt_pts_seg_data)
        >>> print(data_sample)
        <Det3DDataSample(
            META INFORMATION
            DATA FIELDS
            gt_pts_seg: <PointData(
                    META INFORMATION
                    DATA FIELDS
                    pts_semantic_mask: tensor([0.7199, 0.4006])
                    pts_instance_mask: tensor([0.7363, 0.8096])
                ) at 0x7f7e2962cc40>
        ) at 0x7f7e29ff0d60>
    """  # noqa: E501

    @property
    def gt_instances_3d(self) -> InstanceData:
        return self._gt_instances_3d

    @gt_instances_3d.setter
    def gt_instances_3d(self, value: InstanceData) -> None:
        self.set_field(value, "_gt_instances_3d", dtype=InstanceData)

    @gt_instances_3d.deleter
    def gt_instances_3d(self) -> None:
        del self._gt_instances_3d

    @property
    def pred_instances_3d(self) -> InstanceData:
        return self._pred_instances_3d

    @pred_instances_3d.setter
    def pred_instances_3d(self, value: InstanceData) -> None:
        self.set_field(value, "_pred_instances_3d", dtype=InstanceData)

    @pred_instances_3d.deleter
    def pred_instances_3d(self) -> None:
        del self._pred_instances_3d

    @property
    def pts_pred_instances_3d(self) -> InstanceData:
        return self._pts_pred_instances_3d

    @pts_pred_instances_3d.setter
    def pts_pred_instances_3d(self, value: InstanceData) -> None:
        self.set_field(value, "_pts_pred_instances_3d", dtype=InstanceData)

    @pts_pred_instances_3d.deleter
    def pts_pred_instances_3d(self) -> None:
        del self._pts_pred_instances_3d

    @property
    def img_pred_instances_3d(self) -> InstanceData:
        return self._img_pred_instances_3d

    @img_pred_instances_3d.setter
    def img_pred_instances_3d(self, value: InstanceData) -> None:
        self.set_field(value, "_img_pred_instances_3d", dtype=InstanceData)

    @img_pred_instances_3d.deleter
    def img_pred_instances_3d(self) -> None:
        del self._img_pred_instances_3d

    @property
    def gt_pts_seg(self) -> PointData:
        return self._gt_pts_seg

    @gt_pts_seg.setter
    def gt_pts_seg(self, value: PointData) -> None:
        self.set_field(value, "_gt_pts_seg", dtype=PointData)

    @gt_pts_seg.deleter
    def gt_pts_seg(self) -> None:
        del self._gt_pts_seg

    @property
    def pred_pts_seg(self) -> PointData:
        return self._pred_pts_seg

    @pred_pts_seg.setter
    def pred_pts_seg(self, value: PointData) -> None:
        self.set_field(value, "_pred_pts_seg", dtype=PointData)

    @pred_pts_seg.deleter
    def pred_pts_seg(self) -> None:
        del self._pred_pts_seg


def to_tensor(data: Union[torch.Tensor, np.ndarray, Sequence, int, float]) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: the converted data.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        if data.dtype is dtype("float64"):
            data = data.astype(np.float32)
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmengine.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f"type {type(data)} cannot be converted to tensor.")


class Pack3DDetInputs(BaseTransform):
    INPUTS_KEYS = ["points", "img"]
    INSTANCEDATA_3D_KEYS = ["gt_bboxes_3d", "gt_labels_3d", "attr_labels", "depths", "centers_2d"]
    INSTANCEDATA_2D_KEYS = [
        "gt_bboxes",
        "gt_bboxes_labels",
    ]

    SEG_KEYS = ["gt_seg_map", "pts_instance_mask", "pts_semantic_mask", "gt_semantic_seg"]

    def __init__(
        self,
        keys: tuple,
        meta_keys: tuple = (
            "img_path",
            "ori_shape",
            "img_shape",
            "lidar2img",
            "depth2img",
            "cam2img",
            "pad_shape",
            "scale_factor",
            "flip",
            "pcd_horizontal_flip",
            "pcd_vertical_flip",
            "box_mode_3d",
            "box_type_3d",
            "img_norm_cfg",
            "num_pts_feats",
            "pcd_trans",
            "sample_idx",
            "pcd_scale_factor",
            "pcd_rotation",
            "pcd_rotation_angle",
            "lidar_path",
            "transformation_3d_flow",
            "trans_mat",
            "affine_aug",
            "sweep_img_metas",
            "ori_cam2img",
            "cam2global",
            "crop_offset",
            "img_crop_offset",
            "resize_img_shape",
            "lidar2cam",
            "ori_lidar2img",
            "num_ref_frames",
            "num_views",
            "ego2global",
            "axis_align_matrix",
        ),
    ) -> None:
        self.keys = keys
        self.meta_keys = meta_keys

    def _remove_prefix(self, key: str) -> str:
        if key.startswith("gt_"):
            key = key[3:]
        return key

    def transform(self, results: Union[dict, List[dict]]) -> Union[dict, List[dict]]:
        """Method to pack the input data. when the value in this dict is a
        list, it usually is in Augmentations Testing.

        Args:
            results (dict | list[dict]): Result dict from the data pipeline.

        Returns:
            dict | List[dict]:

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info of
              the sample.
        """
        # augtest
        if isinstance(results, list):
            if len(results) == 1:
                # simple test
                return self.pack_single_results(results[0])
            pack_results = []
            for single_result in results:
                pack_results.append(self.pack_single_results(single_result))
            return pack_results
        # norm training and simple testing
        elif isinstance(results, dict):
            return self.pack_single_results(results)
        else:
            raise NotImplementedError

    def pack_single_results(self, results: dict) -> dict:
        """Method to pack the single input data. when the value in this dict is
        a list, it usually is in Augmentations Testing.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info
              of the sample.
        """
        # Format 3D data
        if "points" in results:
            if isinstance(results["points"], BasePoints):
                results["points"] = results["points"].tensor

        if "img" in results:
            if isinstance(results["img"], list):
                # process multiple imgs in single frame
                imgs = np.stack(results["img"], axis=0)
                if imgs.flags.c_contiguous:
                    imgs = to_tensor(imgs).permute(0, 3, 1, 2).contiguous()
                else:
                    imgs = to_tensor(np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
                results["img"] = imgs
            else:
                img = results["img"]
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                # To improve the computational speed by by 3-5 times, apply:
                # `torch.permute()` rather than `np.transpose()`.
                # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
                # for more details
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                else:
                    img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
                results["img"] = img

        for key in [
            "proposals",
            "gt_bboxes",
            "gt_bboxes_ignore",
            "gt_labels",
            "gt_bboxes_labels",
            "attr_labels",
            "pts_instance_mask",
            "pts_semantic_mask",
            "centers_2d",
            "depths",
            "gt_labels_3d",
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]
            else:
                results[key] = to_tensor(results[key])
        if "gt_bboxes_3d" in results:
            if not isinstance(results["gt_bboxes_3d"], BaseInstance3DBoxes):
                results["gt_bboxes_3d"] = to_tensor(results["gt_bboxes_3d"])

        if "gt_semantic_seg" in results:
            results["gt_semantic_seg"] = to_tensor(results["gt_semantic_seg"][None])
        if "gt_seg_map" in results:
            results["gt_seg_map"] = results["gt_seg_map"][None, ...]

        data_sample = Det3DDataSample()
        gt_instances_3d = InstanceData()
        gt_instances = InstanceData()
        gt_pts_seg = PointData()

        data_metas = {}
        for key in self.meta_keys:
            if key in results:
                data_metas[key] = results[key]
            elif "images" in results:
                if len(results["images"].keys()) == 1:
                    cam_type = list(results["images"].keys())[0]
                    # single-view image
                    if key in results["images"][cam_type]:
                        data_metas[key] = results["images"][cam_type][key]
                else:
                    # multi-view image
                    img_metas = []
                    cam_types = list(results["images"].keys())
                    for cam_type in cam_types:
                        if key in results["images"][cam_type]:
                            img_metas.append(results["images"][cam_type][key])
                    if len(img_metas) > 0:
                        data_metas[key] = img_metas
            elif "lidar_points" in results:
                if key in results["lidar_points"]:
                    data_metas[key] = results["lidar_points"][key]
        data_sample.set_metainfo(data_metas)

        inputs = {}
        for key in self.keys:
            if key in results:
                if key in self.INPUTS_KEYS:
                    inputs[key] = results[key]
                elif key in self.INSTANCEDATA_3D_KEYS:
                    gt_instances_3d[self._remove_prefix(key)] = results[key]
                elif key in self.INSTANCEDATA_2D_KEYS:
                    if key == "gt_bboxes_labels":
                        gt_instances["labels"] = results[key]
                    else:
                        gt_instances[self._remove_prefix(key)] = results[key]
                elif key in self.SEG_KEYS:
                    gt_pts_seg[self._remove_prefix(key)] = results[key]
                else:
                    raise NotImplementedError(
                        f"Please modified " f"`Pack3DDetInputs` " f"to put {key} to " f"corresponding field"
                    )

        data_sample.gt_instances_3d = gt_instances_3d
        data_sample.gt_instances = gt_instances
        data_sample.gt_pts_seg = gt_pts_seg
        if "eval_ann_info" in results:
            data_sample.eval_ann_info = results["eval_ann_info"]
        else:
            data_sample.eval_ann_info = None

        packed_results = dict()
        packed_results["data_samples"] = data_sample
        packed_results["inputs"] = inputs

        return packed_results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(keys={self.keys})"
        repr_str += f"(meta_keys={self.meta_keys})"
        return repr_str


class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict, callable], optional): Sequence of transform
            object or config dict to be composed.
    """

    def __init__(self, transforms: Optional[Sequence[Union[dict, Callable]]]):
        self.transforms: List[Callable] = []

        if transforms is None:
            transforms = []

        for transform in transforms:
            # `Compose` can be built with config dict with type and
            # corresponding arguments.
            if isinstance(transform, dict):
                print("transform", transform)
                if transform["type"] == "LidarDet3DInferencerLoader":
                    transform.pop("type", None)
                    print("transform here", transform)
                    transform = LidarDet3DInferencerLoader(**transform)
                elif transform["type"] == "LoadPointsFromMultiSweeps":
                    transform.pop("type", None)
                    print("transform here", transform)
                    transform = LoadPointsFromMultiSweeps(**transform)
                elif transform["type"] == "MultiScaleFlipAug3D":
                    transform.pop("type", None)
                    print("transform here", transform)
                    transform = MultiScaleFlipAug3D(**transform)
                elif transform["type"] == "GlobalRotScaleTrans":
                    transform.pop("type", None)
                    print("transform here", transform)
                    transform = GlobalRotScaleTrans(**transform)
                elif transform["type"] == "Pack3DDetInputs":
                    transform.pop("type", None)
                    print("transform here", transform)
                    transform = Pack3DDetInputs(**transform)
                elif transform["type"] == "RandomFlip3D":
                    transform.pop("type", None)
                    print("transform here", transform)
                    transform = RandomFlip3D(**transform)
                elif transform["type"] == "PointsRangeFilter":
                    transform.pop("type", None)
                    print("transform here", transform)
                    transform = PointsRangeFilter(**transform)
                # elif transform["type"]=="GlobalRotScaleTrans":
                #     transform.pop('type', None)
                #     print("transform here",transform)
                #     transform = MultiScaleFlipAug3D(**GlobalRotScaleTrans)
                # if not callable(transform):
                #     raise TypeError(f'transform should be a callable object, '
                #                     f'but got {type(transform)}')
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(f"transform must be a callable object or dict, " f"but got {type(transform)}")

    def __call__(self, data: dict) -> Optional[dict]:
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            print("t", t)
            data = t(data)
            # The transform will return None when it failed to load images or
            # cannot find suitable augmentation parameters to augment the data.
            # Here we simply return None if the transform returns None and the
            # dataset will handle it by randomly selecting another data sample.
            if data is None:
                return None
        return data

    def __repr__(self):
        """Print ``self.transforms`` in sequence.

        Returns:
            str: Formatted string.
        """
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
