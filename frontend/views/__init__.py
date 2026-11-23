"""Views package for FreshCheck AI."""

from frontend.views.analyzer import render_analyzer_view
from frontend.views.dashboard import render_dashboard_view
from frontend.views.history import render_history_view
from frontend.views.results import render_results_view

__all__ = [
    "render_dashboard_view",
    "render_analyzer_view",
    "render_results_view",
    "render_history_view",
]
