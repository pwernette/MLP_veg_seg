from qgis.PyQt.QtWidgets import QAction, QFileDialog, QMessageBox
from qgis.core import QgsProject
from .ml_runner import run_reclassification
from .ui_ml_veg_reclass import Ui_MLReclassDialog
from PyQt5.QtWidgets import QDialog

class MLVegReclassPlugin:
    def __init__(self, iface):
        self.iface = iface

    def initGui(self):
        self.action = QAction("Vegetation Reclassifier", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addPluginToMenu("&Vegetation Reclassifier", self.action)

    def unload(self):
        self.iface.removePluginMenu("&Vegetation Reclassifier", self.action)

    def run(self):
        dialog = QDialog()
        ui = Ui_MLReclassDialog()
        ui.setupUi(dialog)

        def on_run_clicked():
            input_path = ui.inputFile.text()
            model_path = ui.modelFile.text()
            output_path = ui.outputFile.text()

            stdout, stderr = run_reclassification(input_path, model_path, output_path)
            if stderr:
                QMessageBox.critical(dialog, "Error", stderr)
            else:
                QMessageBox.information(dialog, "Success", "Reclassification complete!")

        ui.runButton.clicked.connect(on_run_clicked)
        dialog.exec_()