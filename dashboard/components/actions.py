"""Action buttons, interactive triggers, and table formatting helpers."""

from typing import Any, List, Optional
import streamlit as st


def primary_button(
    label: str,
    key: Optional[str] = None,
    help: Optional[str] = None,
    disabled: bool = False,
    use_container_width: bool = False,
) -> bool:
    """Render a primary call-to-action button.

    Args:
        label: Button text.
        key: Streamlit widget key.
        help: Tooltip text.
        disabled: Disabled state.
        use_container_width: Expand to container width.

    Returns:
        bool: True if clicked.
    """
    return st.button(
        label=label,
        key=key,
        help=help,
        type="primary",
        disabled=disabled,
        use_container_width=use_container_width,
    )


def secondary_button(
    label: str,
    key: Optional[str] = None,
    help: Optional[str] = None,
    disabled: bool = False,
    use_container_width: bool = False,
) -> bool:
    """Render a secondary action button.

    Args:
        label: Button text.
        key: Streamlit widget key.
        help: Tooltip text.
        disabled: Disabled state.
        use_container_width: Expand to container width.

    Returns:
        bool: True if clicked.
    """
    return st.button(
        label=label,
        key=key,
        help=help,
        type="secondary",
        disabled=disabled,
        use_container_width=use_container_width,
    )


def styled_table(
    data: Any,
    column_config: Optional[dict] = None,
    use_container_width: bool = True,
    hide_index: bool = True,
) -> None:
    """Render a dataframe or record dictionary using native styled dataframe table.

    Args:
        data: Tabular data (DataFrame, list of dicts).
        column_config: Streamlit column configuration dictionary.
        use_container_width: Expand to fit container.
        hide_index: Whether to suppress row indices.
    """
    try:
        st.dataframe(
            data,
            column_config=column_config,
            width="stretch" if use_container_width else "content",
            hide_index=hide_index,
        )
    except TypeError:
        st.dataframe(
            data,
            column_config=column_config,
            use_container_width=use_container_width,
            hide_index=hide_index,
        )
