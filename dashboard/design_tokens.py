"""Centralized Design System Tokens for the Cybersecurity NIDS Platform.

This module acts as the single source of truth for color palettes, typography,
spacing, borders, elevation, and component sizing across the dashboard UI.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class ColorPalette:
    """Core color tokens adhering to the modern enterprise cybersecurity dark theme."""

    # Canvas & Surface
    background: str = "#0B1020"
    secondary_background: str = "#111827"
    surface: str = "#172033"
    surface_elevated: str = "#1E293B"
    surface_hover: str = "#223147"

    # Brand & Accent
    primary_accent: str = "#6385FF"
    primary_hover: str = "#4F70E6"
    primary_muted: str = "rgba(99, 133, 255, 0.15)"

    secondary_accent: str = "#8B5CF6"
    secondary_hover: str = "#7C3AED"
    secondary_muted: str = "rgba(139, 92, 246, 0.15)"

    # Text & Typography
    text_primary: str = "#F1F5F9"
    text_secondary: str = "#94A3B8"
    text_muted: str = "#64748B"
    text_inverse: str = "#0B1020"

    # Borders & Dividers
    border_subtle: str = "#1E293B"
    border_default: str = "#29364B"
    border_accent: str = "#3D4F6C"
    border_highlight: str = "#6385FF"

    # Semantic Status Indicators
    status_success: str = "#10B981"
    status_success_bg: str = "rgba(16, 185, 129, 0.12)"
    status_success_border: str = "rgba(16, 185, 129, 0.35)"

    status_warning: str = "#F59E0B"
    status_warning_bg: str = "rgba(245, 158, 11, 0.12)"
    status_warning_border: str = "rgba(245, 158, 11, 0.35)"

    status_error: str = "#EF4444"
    status_error_bg: str = "rgba(239, 68, 68, 0.12)"
    status_error_border: str = "rgba(239, 68, 68, 0.35)"

    status_info: str = "#3B82F6"
    status_info_bg: str = "rgba(59, 130, 246, 0.12)"
    status_info_border: str = "rgba(59, 130, 246, 0.35)"

    status_neutral: str = "#64748B"
    status_neutral_bg: str = "rgba(100, 116, 139, 0.12)"
    status_neutral_border: str = "rgba(100, 116, 139, 0.35)"


@dataclass(frozen=True)
class TypographyTokens:
    """Typography scales, weights, and font family definitions."""

    font_family_sans: str = (
        "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', "
        "'Helvetica Neue', Arial, sans-serif"
    )
    font_family_mono: str = (
        "'JetBrains Mono', 'Fira Code', 'Cascadia Code', Consolas, monospace"
    )

    size_xs: str = "0.75rem"      # 12px
    size_sm: str = "0.875rem"     # 14px
    size_base: str = "1.000rem"   # 16px
    size_lg: str = "1.125rem"     # 18px
    size_xl: str = "1.250rem"     # 20px
    size_2xl: str = "1.500rem"    # 24px
    size_3xl: str = "1.875rem"    # 30px

    weight_normal: int = 400
    weight_medium: int = 500
    weight_semibold: int = 600
    weight_bold: int = 700


@dataclass(frozen=True)
class SpacingTokens:
    """Spacing dimensions for margin, padding, and layout grid."""

    xs: str = "0.25rem"   # 4px
    sm: str = "0.50rem"   # 8px
    md: str = "1.00rem"   # 16px
    lg: str = "1.50rem"   # 24px
    xl: str = "2.00rem"   # 32px
    xxl: str = "3.00rem"  # 48px


@dataclass(frozen=True)
class RadiusTokens:
    """Corner radii for cards, buttons, badges, and modal dialogs."""

    sm: str = "4px"
    md: str = "8px"
    lg: str = "12px"
    xl: str = "16px"
    full: str = "9999px"


@dataclass(frozen=True)
class ShadowTokens:
    """Depth and elevation shadow definitions."""

    subtle: str = "0 1px 2px 0 rgba(0, 0, 0, 0.35)"
    card: str = "0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -2px rgba(0, 0, 0, 0.3)"
    elevated: str = "0 10px 15px -3px rgba(0, 0, 0, 0.5), 0 4px 6px -4px rgba(0, 0, 0, 0.4)"


@dataclass(frozen=True)
class DesignTokens:
    """Master Design Tokens Container."""

    colors: ColorPalette = field(default_factory=ColorPalette)
    typography: TypographyTokens = field(default_factory=TypographyTokens)
    spacing: SpacingTokens = field(default_factory=SpacingTokens)
    radius: RadiusTokens = field(default_factory=RadiusTokens)
    shadows: ShadowTokens = field(default_factory=ShadowTokens)

    def to_css_variables(self) -> str:
        """Export tokens as CSS custom properties for stylesheet consumption."""
        return f"""
:root {{
    --color-bg: {self.colors.background};
    --color-bg-secondary: {self.colors.secondary_background};
    --color-surface: {self.colors.surface};
    --color-surface-elevated: {self.colors.surface_elevated};
    --color-surface-hover: {self.colors.surface_hover};

    --color-accent-primary: {self.colors.primary_accent};
    --color-accent-primary-hover: {self.colors.primary_hover};
    --color-accent-primary-muted: {self.colors.primary_muted};

    --color-accent-secondary: {self.colors.secondary_accent};
    --color-accent-secondary-hover: {self.colors.secondary_hover};
    --color-accent-secondary-muted: {self.colors.secondary_muted};

    --color-text-primary: {self.colors.text_primary};
    --color-text-secondary: {self.colors.text_secondary};
    --color-text-muted: {self.colors.text_muted};

    --color-border-subtle: {self.colors.border_subtle};
    --color-border-default: {self.colors.border_default};
    --color-border-accent: {self.colors.border_accent};

    --color-status-success: {self.colors.status_success};
    --color-status-success-bg: {self.colors.status_success_bg};
    --color-status-success-border: {self.colors.status_success_border};

    --color-status-warning: {self.colors.status_warning};
    --color-status-warning-bg: {self.colors.status_warning_bg};
    --color-status-warning-border: {self.colors.status_warning_border};

    --color-status-error: {self.colors.status_error};
    --color-status-error-bg: {self.colors.status_error_bg};
    --color-status-error-border: {self.colors.status_error_border};

    --color-status-info: {self.colors.status_info};
    --color-status-info-bg: {self.colors.status_info_bg};
    --color-status-info-border: {self.colors.status_info_border};

    --color-status-neutral: {self.colors.status_neutral};
    --color-status-neutral-bg: {self.colors.status_neutral_bg};
    --color-status-neutral-border: {self.colors.status_neutral_border};

    --font-sans: {self.typography.font_family_sans};
    --font-mono: {self.typography.font_family_mono};

    --radius-sm: {self.radius.sm};
    --radius-md: {self.radius.md};
    --radius-lg: {self.radius.lg};
    --radius-xl: {self.radius.xl};
    --radius-full: {self.radius.full};

    --shadow-subtle: {self.shadows.subtle};
    --shadow-card: {self.shadows.card};
    --shadow-elevated: {self.shadows.elevated};
}}
""".strip()


# Global Singleton
tokens = DesignTokens()
