import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from frontend.gui import MainWindow


class MainWindowProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()

    def tearDown(self):
        self.window.close()

    def test_professional_layout_keeps_controls_in_analysis_tabs(self):
        self.window.show()
        self.app.processEvents()
        self.assertEqual(self.window.analysis_tabs.count(), 3)
        self.assertEqual(
            [self.window.analysis_tabs.tabText(i) for i in range(3)],
            ["Measure", "Traces", "Markers"],
        )
        tab_bar = self.window.analysis_tabs.tabBar()
        widths = [tab_bar.tabRect(i).width() for i in range(tab_bar.count())]
        self.assertLessEqual(max(widths) - min(widths), 1)

    def test_side_panels_collapse_and_restore_from_persistent_edge_handles(self):
        self.window.show()
        self.app.processEvents()
        initial_central_width = self.window.centralWidget().width()

        self.window.btn_left_panel_toggle.click()
        self.window.btn_right_panel_toggle.click()
        self.app.processEvents()

        self.assertFalse(self.window.left_dock.isVisible())
        self.assertFalse(self.window.right_dock.isVisible())
        self.assertTrue(self.window.btn_left_panel_toggle.isVisible())
        self.assertTrue(self.window.btn_right_panel_toggle.isVisible())
        self.assertEqual(self.window.btn_left_panel_toggle.text(), "›")
        self.assertEqual(self.window.btn_right_panel_toggle.text(), "‹")
        self.assertEqual(self.window.btn_left_panel_toggle.x(), 0)
        self.assertEqual(
            self.window.btn_right_panel_toggle.geometry().right(),
            self.window.width() - 1,
        )
        self.assertGreater(self.window.centralWidget().width(), initial_central_width)

        self.window.btn_left_panel_toggle.click()
        self.window.btn_right_panel_toggle.click()
        self.app.processEvents()
        self.assertTrue(self.window.left_dock.isVisible())
        self.assertTrue(self.window.right_dock.isVisible())
        self.assertEqual(self.window.btn_left_panel_toggle.text(), "‹")
        self.assertEqual(self.window.btn_right_panel_toggle.text(), "›")

    def test_none_is_default_and_disables_marker_placement(self):
        self.assertEqual(self.window.marker_selector_btn.current_marker_id(), 0)
        self.assertFalse(self.window.btn_delta_marker.isEnabled())
        self.assertIsNone(self.window.spectrum_widget._active_marker_id)

    def test_x301_and_hackrf_profiles_restore_documented_limits(self):
        self.window.sdr_type_combo.setCurrentIndex(
            self.window.sdr_type_combo.findData("USRP")
        )
        usrp = self.window._acquisition_config()
        self.assertEqual(usrp.sample_rate, 200e6)
        self.assertEqual(usrp.span, 160e6)
        self.assertEqual(self.window.span_ctrl._max_hz, 160e6)
        self.assertEqual(self.window.sample_rate_combo.itemText(
            self.window.sample_rate_combo.count() - 1
        ), "200")
        self.window.sample_rate_combo.setCurrentText("100")
        self.assertEqual(self.window.span_ctrl._max_hz, 100e6)
        self.assertEqual(self.window._acquisition_config().span, 100e6)

        self.window.sdr_type_combo.setCurrentIndex(
            self.window.sdr_type_combo.findData("HACKRF")
        )
        hackrf = self.window._acquisition_config()
        self.assertEqual(hackrf.sample_rate, 20e6)
        self.assertEqual(hackrf.span, 20e6)
        self.assertEqual(self.window.span_ctrl._max_hz, 20e6)
        self.assertEqual(self.window.sample_rate_combo.itemText(
            self.window.sample_rate_combo.count() - 1
        ), "20")


if __name__ == "__main__":
    unittest.main()
