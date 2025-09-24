"""Provides a utility function to generate a detailed summary of a PyTorch model's architecture."""

from collections.abc import Sequence
from typing import Any, Literal, overload

import torch
from torch import nn
from torchview import draw_graph
from torchview.computation_graph import compact_list_repr
from torchview.computation_node import FunctionNode, ModuleNode, TensorNode

# Type definition for input_size, consistent with torchview
INPUT_SIZE_TYPE = Sequence[int | Sequence[Any] | torch.Size]


def _build_summary_hierarchy(
    node: dict[ModuleNode, list[TensorNode]] | ModuleNode | FunctionNode | TensorNode,
    indent_level: int = 0,
    summary_lines: list[str] = [],
) -> None:
    """Print a formatted summary of each layer.

    It recursively traverses the node hierarchy from torchview's ComputationGraph
    and prints a formatted summary of each layer.

    Args:
        node (dict[ModuleNode, list[TensorNode]] | ModuleNode | FunctionNode | TensorNode): The
            current node in the hierarchy, which can be a ModuleNode, FunctionNode, or a
            dictionary of ModuleNodes with their children.
        summary_lines (list[str]): A list to which the summary string for each layer is appended.
        indent_level (int): The current indentation level for pretty-printing.
    """
    # Set the indentation for the current level
    indent = "  " * indent_level

    # Case 1: Node is a dictionary representing a Module with children
    # The structure is {ModuleNode: [child1, child2, ...]}
    if isinstance(node, dict):
        module_node, children = next(iter(node.items()))

        # Don't print the top-level dummy container module (depth < 0)
        if module_node.depth >= 0:
            _build_summary_node_info(module_node, indent, summary_lines)

        # Recursively call this function for all children of the module
        for child in children:
            _build_summary_hierarchy(child, indent_level + 1, summary_lines)

    # Case 2: Node is a leaf in the hierarchy (e.g., a function call)
    elif isinstance(node, (FunctionNode, TensorNode)):
        _build_summary_node_info(node, indent, summary_lines)


def _build_summary_node_info(
    node: ModuleNode | FunctionNode | TensorNode, indent: str, summary_lines: list[str]
) -> None:
    """Format and print the information for a single ModuleNode or FunctionNode.

    Args:
        node (ModuleNode | FunctionNode | TensorNode): The node to print information for.
        indent (str): The indentation string for pretty-printing.
        summary_lines (list[str]): The list to append the formatted summary line to.
    """
    # We are only interested in Modules and Functions, not intermediate Tensors
    if not isinstance(node, (ModuleNode, FunctionNode)):
        return

    # Skip printing the "empty-pass" nodes torchview inserts for clarity
    if node.name == "empty-pass":
        return

    # Get input and output shapes, using compact_list_repr for clean formatting
    input_shapes = node.input_shape if hasattr(node, "input_shape") else []
    output_shapes = node.output_shape if hasattr(node, "output_shape") else []

    input_str = compact_list_repr(input_shapes) if input_shapes else "N/A"
    output_str = compact_list_repr(output_shapes) if output_shapes else "N/A"

    summary_lines.append(f"{indent}{node.name}: Input: {input_str} -> Output: {output_str}")


@overload
def model_summary(
    model: nn.Module,
    input_size: INPUT_SIZE_TYPE,
    device: str = "cpu",
    depth: int = 5,
    *,
    hide_module_functions: bool = False,
    return_as_string: Literal[True],
) -> str: ...


@overload
def model_summary(
    model: nn.Module,
    input_size: INPUT_SIZE_TYPE,
    device: str = "cpu",
    depth: int = 5,
    *,
    hide_module_functions: bool = False,
    return_as_string: Literal[False] = False,
) -> None: ...


def model_summary(
    model: nn.Module,
    input_size: INPUT_SIZE_TYPE,
    device: str = "cpu",
    depth: int = 5,
    *,
    hide_module_functions: bool = False,
    return_as_string: bool = False,
) -> str | None:
    """Generate a layer-by-layer summary of a PyTorch model's input and output shapes.

    This function uses torchview to trace the model and then traverses the resulting
    computation graph to print a structured text summary.

    Args:
        model (nn.Module): The PyTorch model to summarize.
        input_size (Sequence of Sizes):
            Shape of input data, e.g., [(1, 3, 224, 224)].
            Use a list of tuples for multiple inputs.
        device (str): The device to run the model on ('cpu' or 'cuda').
        depth (int): The maximum depth of nested modules to display.
        hide_module_functions (bool): If False, shows functions within modules (e.g., the 'conv2d'
            inside an nn.Conv2d). Defaults to False for a more detailed summary.
        return_as_string (bool): If True, the function returns the summary as a string
            instead of printing it to the console. Defaults to False.
    """
    summary_lines = []
    model_name = model.__class__.__name__
    if hasattr(model, "name"):
        model_name += f" ({model.name})"

    summary_lines.append(f"Model Summary: {model_name}")
    summary_lines.append("-" * 80)

    # Use draw_graph to perform the forward pass and build the computation graph.
    # We set save_graph=False as we only need the returned graph object.
    graph = draw_graph(
        model,
        input_size=input_size,
        device=device,
        save_graph=False,
        # Set parameters to get the desired level of detail for the summary
        hide_module_functions=hide_module_functions,
        depth=depth,
    )

    # The 'node_hierarchy' attribute contains the full, structured graph.
    # We pass it to our recursive printing function to generate the summary.
    _build_summary_hierarchy(graph.node_hierarchy, 0, summary_lines)

    summary_lines.append("-" * 80)

    result = "\n".join(summary_lines)

    if return_as_string:
        return result

    print(result)
    return None
