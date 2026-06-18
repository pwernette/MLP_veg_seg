# -*- coding: utf-8 -*-
"""
Shared base class, parameter names, and helper methods used by all three
ML Vegetation Filter Processing algorithms.
"""
import os
import sys

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterString,
)


# ---------------------------------------------------------------------------
# Vegetation-index preset descriptions shown in the UI drop-down
# ---------------------------------------------------------------------------
VEG_INDEX_OPTIONS = [
    'rgb       – normalised R, G, B only (no derived indices)',
    'simple    – R, G, B + ExR, ExG, ExB, ExGR',
    'all       – all 10 RGB-derived vegetation indices',
    'custom    – enter individual index names in the field below',
]
VEG_INDEX_KEYS = ['rgb', 'simple', 'all', 'custom']

CUSTOM_INDEX_HELP = (
    'Comma-separated list of individual indices to use when "custom" is selected '
    'above. Available tokens: exr, exg, exb, exgr, ngrdi, mgrvi, gli, rgbvi, '
    'ikaw, gla, xyz (normalised coords), sd (per-axis std dev), 3d (3-D std dev). '
    'Example: r,g,b,gli,ngrdi'
)


def _plugin_root():
    """Absolute path to the plugin root directory (contains src/)."""
    return os.path.dirname(os.path.dirname(__file__))


def _ensure_src_on_path():
    """
    Insert the plugin root onto sys.path so that  'from src.fileio import …'
    works inside the algorithm processAlgorithm() methods.
    Idempotent – safe to call multiple times.
    """
    root = _plugin_root()
    if root not in sys.path:
        sys.path.insert(0, root)


class MLVegBaseAlgorithm(QgsProcessingAlgorithm):
    """
    Shared parameter names and helper methods.

    Subclasses call the _add_*() helpers inside initAlgorithm() to avoid
    repeating the same parameter definitions across all three algorithms.
    """

    # ---- shared parameter name constants ----
    P_VEG_INDEX      = 'VEG_INDEX'
    P_CUSTOM_INDICES = 'CUSTOM_INDICES'
    P_MODEL_NODES    = 'MODEL_NODES'
    P_MODEL_DROPOUT  = 'MODEL_DROPOUT'
    P_GEOM_RADIUS    = 'GEOM_RADIUS'

    P_EPOCHS         = 'EPOCHS'
    P_BATCH_SIZE     = 'BATCH_SIZE'
    P_SHUFFLE        = 'SHUFFLE'
    P_CACHE          = 'CACHE'
    P_PREFETCH       = 'PREFETCH'
    P_SPLIT          = 'SPLIT'
    P_CLASS_IMBAL    = 'CLASS_IMBALANCE'
    P_DATA_REDUCTION = 'DATA_REDUCTION'
    P_EARLY_PATIENCE = 'EARLY_PATIENCE'
    P_EARLY_DELTA    = 'EARLY_DELTA'
    P_VERBOSE        = 'VERBOSE'

    # ------------------------------------------------------------------
    # Parameter-group helpers
    # ------------------------------------------------------------------

    def _add_veg_index_params(self):
        """Add vegetation-index preset drop-down + custom field."""
        self.addParameter(
            QgsProcessingParameterEnum(
                self.P_VEG_INDEX,
                'Vegetation index preset',
                options=VEG_INDEX_OPTIONS,
                defaultValue=0,
            )
        )
        p = QgsProcessingParameterString(
            self.P_CUSTOM_INDICES,
            'Custom indices (comma-separated; only used when "custom" is selected)',
            defaultValue='r,g,b,exg,exr',
            optional=True,
        )
        p.setHelp(CUSTOM_INDEX_HELP)
        self.addParameter(p)

    def _add_model_arch_params(self):
        """Add model architecture parameters (nodes, dropout, geometry radius)."""
        self.addParameter(
            QgsProcessingParameterString(
                self.P_MODEL_NODES,
                'Hidden layer nodes (comma-separated, one value per layer)',
                defaultValue='8,8,8',
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.P_MODEL_DROPOUT,
                'Dropout rate (0.0 – 0.99)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.2,
                minValue=0.0,
                maxValue=0.99,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.P_GEOM_RADIUS,
                'Geometry search radius in metres (only used when sd or 3d index is selected)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.10,
                minValue=0.001,
            )
        )

    def _add_training_params(self):
        """Add all training hyperparameter controls."""
        self.addParameter(
            QgsProcessingParameterNumber(
                self.P_EPOCHS,
                'Maximum training epochs',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=100,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.P_BATCH_SIZE,
                'Batch size',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1000,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.P_SHUFFLE, 'Shuffle training data', defaultValue=True
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.P_CACHE, 'Cache dataset in memory (speeds up training)', defaultValue=True
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.P_PREFETCH, 'Prefetch batches (speeds up training)', defaultValue=True
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.P_SPLIT,
                'Training / validation split proportion (0.01 – 0.99)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.7,
                minValue=0.01,
                maxValue=0.99,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.P_CLASS_IMBAL,
                'Correct for class imbalance (subsample the larger class)',
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.P_DATA_REDUCTION,
                'Data reduction factor (1.0 = use all data, 0.5 = use 50%, etc.)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.0,
                minValue=0.01,
                maxValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.P_EARLY_PATIENCE,
                'Early stopping patience (epochs without improvement before stopping)',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=10,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.P_EARLY_DELTA,
                'Early stopping minimum delta (minimum improvement to continue)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.01,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.P_VERBOSE,
                'TensorFlow verbosity level (0 = silent, 1 = progress bar, 2 = full)',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=2,
                minValue=0,
                maxValue=2,
            )
        )

    # ------------------------------------------------------------------
    # Value-extraction helpers used inside processAlgorithm()
    # ------------------------------------------------------------------

    def _resolve_veg_indices(self, parameters, context):
        """
        Return (veg_indices, model_inputs) based on the drop-down selection.
        veg_indices is the string/list expected by las2split() / vegidx().
        model_inputs is the matching list of column names.
        """
        idx_choice = self.parameterAsEnum(parameters, self.P_VEG_INDEX, context)
        key = VEG_INDEX_KEYS[idx_choice]

        if key == 'rgb':
            return 'rgb', ['r', 'g', 'b']

        if key == 'simple':
            return 'simple', ['r', 'g', 'b', 'exr', 'exg', 'exb', 'exgr']

        if key == 'all':
            return 'all', [
                'r', 'g', 'b',
                'exr', 'exg', 'exb', 'exgr',
                'ngrdi', 'mgrvi', 'gli', 'rgbvi', 'ikaw', 'gla',
            ]

        # custom
        raw = self.parameterAsString(parameters, self.P_CUSTOM_INDICES, context)
        tokens = [t.strip() for t in raw.split(',') if t.strip()]
        return tokens, tokens

    def _resolve_nodes(self, parameters, context):
        """Parse the comma-separated node string into a list of ints."""
        raw = self.parameterAsString(parameters, self.P_MODEL_NODES, context)
        return [int(x.strip()) for x in raw.split(',') if x.strip()]

    def _geom_metrics_from_indices(self, veg_indices):
        """
        Derive the geometry_metrics list that las2split() expects from the
        chosen vegetation-index preset / custom list.
        """
        if isinstance(veg_indices, list):
            if any('sd' in v for v in veg_indices):
                return ['sd']
            if any('3d' in v for v in veg_indices):
                return ['3d']
        return []
