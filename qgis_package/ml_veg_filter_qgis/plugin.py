# -*- coding: utf-8 -*-
import os
from qgis.core import QgsApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction
from .provider import MLVegFilterProvider


class MLVegFilterPlugin:
    """QGIS Plugin implementation."""

    def __init__(self, iface):
        self.iface = iface
        self.provider = None
        self.action = None

    # ------------------------------------------------------------------
    def initGui(self):
        """Register the Processing provider and add a toolbar shortcut."""
        self.provider = MLVegFilterProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

        icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
        self.action = QAction(
            QIcon(icon_path),
            'ML Vegetation Filter',
            self.iface.mainWindow(),
        )
        self.action.setToolTip(
            'Open ML Vegetation Filter in the Processing Toolbox'
        )
        self.action.triggered.connect(self._open_toolbox)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToMenu('ML Vegetation Filter', self.action)

    def unload(self):
        """Remove the provider and clean up UI elements."""
        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)
        if self.action:
            self.iface.removeToolBarIcon(self.action)
            self.iface.removePluginMenu('ML Vegetation Filter', self.action)

    # ------------------------------------------------------------------
    def _open_toolbox(self):
        """Focus the Processing Toolbox."""
        try:
            from processing.gui.ProcessingToolbox import ProcessingToolbox
            toolbox = self.iface.mainWindow().findChild(ProcessingToolbox)
            if toolbox:
                toolbox.show()
                toolbox.raise_()
        except Exception:
            pass
