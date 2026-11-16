"""Integration tests for all 8 Streamlit navigation routes using Streamlit AppTest."""

from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest

from dashboard.navigation import ROUTES

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


class TestAppNavigationRoutes(unittest.TestCase):
    """End-to-end integration tests cycling through all application routes."""

    def test_all_eight_routes_render_without_exceptions(self) -> None:
        """Verify each of the 8 navigation routes renders cleanly without exceptions."""
        app = AppTest.from_file(APP_PATH, default_timeout=20).run()
        self.assertEqual(len(app.exception), 0, "Initial run on Overview threw exceptions")
        self.assertEqual(app.session_state["nids_active_route"], "overview")

        radio = app.sidebar.radio("nids_navigation_radio")
        for route in ROUTES:
            radio.set_value(route.display_name).run()
            self.assertEqual(
                len(app.exception),
                0,
                f"Route '{route.route_id}' threw exceptions: {app.exception}",
            )
            self.assertEqual(
                app.session_state["nids_active_route"],
                route.route_id,
                f"Session state active route does not match '{route.route_id}'",
            )


if __name__ == "__main__":
    unittest.main()
