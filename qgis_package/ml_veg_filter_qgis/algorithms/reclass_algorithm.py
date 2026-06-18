# -*- coding: utf-8 -*-
"""
QGIS Processing Algorithm: Reclassify point cloud with a saved model
=====================================================================
Loads a trained TensorFlow model (.keras or .h5) and applies it to a
LAS/LAZ point cloud, writing the updated classification (and optionally
per-class probability extra bytes) to a new LAZ file.

Mirrors the logic of ML_veg_reclass.py.
"""
import os
import sys

from qgis.core import (
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFolderDestination,
    QgsProcessingOutputFolder,
    QgsProcessingException,
)

from .base import MLVegBaseAlgorithm, _ensure_src_on_path


class MLVegReclassAlgorithm(MLVegBaseAlgorithm):
    """Reclassify a point cloud using a previously trained model."""

    P_MODEL_FILE   = 'MODEL_FILE'
    P_RECLASS_FILE = 'RECLASS_FILE'
    P_BATCH_SIZE   = 'BATCH_SIZE'
    P_CACHE        = 'CACHE'
    P_WRITE_PROBS  = 'WRITE_PROBS'
    P_OUTPUT_DIR   = 'OUTPUT_DIR'

    # ------------------------------------------------------------------
    def name(self):
        return 'reclass_pointcloud'

    def displayName(self):
        return '2 – Reclassify point cloud with saved model'

    def group(self):
        return 'ML Vegetation Filter'

    def groupId(self):
        return 'mlvegfilter'

    def shortHelpString(self):
        return (
            '<p><b>Apply a previously trained TensorFlow MLP model to a '
            'LAS/LAZ point cloud.</b></p>'
            '<p><b>Model file</b> – Select the <code>.keras</code> or '
            '<code>.h5</code> file produced by the Train algorithm. '
            'The required vegetation indices are inferred automatically from '
            'the model\'s input layer names.</p>'
            '<p><b>Point cloud to reclassify</b> – Any RGB-coloured LAS/LAZ '
            'file. The <code>classification</code> field will be updated '
            'with the model\'s predictions.</p>'
            '<p><b>Output</b> – A new LAZ file named '
            '<code>&lt;original&gt;_&lt;model_name&gt;.laz</code> written to '
            'the chosen output folder. Per-class probabilities are optionally '
            'stored as extra-bytes fields (<code>prob0</code>, '
            '<code>prob1</code>, …).</p>'
            '<p>Based on: Wernette (2024) '
            '<a href="https://doi.org/10.5281/zenodo.10966854">'
            'doi:10.5281/zenodo.10966854</a></p>'
        )

    def createInstance(self):
        return MLVegReclassAlgorithm()

    # ------------------------------------------------------------------
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.P_MODEL_FILE,
                'Trained model file (.keras or .h5)',
                # extension left blank so both .keras and .h5 are selectable
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.P_RECLASS_FILE,
                'Point cloud to reclassify (.las or .laz)',
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.P_BATCH_SIZE,
                'Prediction batch size',
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1000,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.P_CACHE,
                'Cache dataset during prediction',
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.P_WRITE_PROBS,
                'Write per-class probabilities as extra-bytes fields',
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.P_GEOM_RADIUS,
                'Geometry search radius in metres (only used if model was trained with sd or 3d inputs)',
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.10,
                minValue=0.001,
            )
        )
        self.addParameter(
            QgsProcessingParameterFolderDestination(
                self.P_OUTPUT_DIR,
                'Output folder for reclassified LAZ file',
            )
        )
        self.addOutput(
            QgsProcessingOutputFolder(self.P_OUTPUT_DIR, 'Output folder')
        )

    # ------------------------------------------------------------------
    def processAlgorithm(self, parameters, context, feedback):
        _ensure_src_on_path()

        # ---- collect parameters ----------------------------------------
        model_file   = self.parameterAsFile(parameters, self.P_MODEL_FILE, context)
        reclass_file = self.parameterAsFile(parameters, self.P_RECLASS_FILE, context)
        batch_size   = self.parameterAsInt(parameters, self.P_BATCH_SIZE, context)
        cache        = self.parameterAsBoolean(parameters, self.P_CACHE, context)
        write_probs  = self.parameterAsBoolean(parameters, self.P_WRITE_PROBS, context)
        geom_radius  = self.parameterAsDouble(parameters, self.P_GEOM_RADIUS, context)
        output_dir   = self.parameterAsString(parameters, self.P_OUTPUT_DIR, context)

        # ---- validate --------------------------------------------------
        if not os.path.isfile(model_file):
            raise QgsProcessingException(f'Model file not found: {model_file}')
        if not os.path.isfile(reclass_file):
            raise QgsProcessingException(f'Point cloud file not found: {reclass_file}')
        os.makedirs(output_dir, exist_ok=True)

        # ---- imports ---------------------------------------------------
        feedback.pushInfo('Importing TensorFlow backend …')
        try:
            import tensorflow as tf
            from src.fileio import predict_reclass_write
        except ImportError as e:
            raise QgsProcessingException(
                f'Required package not found: {e}\n'
                'Install with: pip install tensorflow laspy lazrs pandas '
                'scikit-learn tqdm'
            )

        # ---- load model ------------------------------------------------
        feedback.pushInfo(f'Loading model: {model_file}')
        try:
            model = tf.keras.models.load_model(model_file)
        except Exception as e:
            raise QgsProcessingException(f'Failed to load model: {e}')

        # Infer required vegetation indices from model input layer names
        model_input_names = [inp.name for inp in model.inputs]
        feedback.pushInfo(f'Model input layers detected: {model_input_names}')

        # geometry metrics
        acceptablegeometrics = ['sd', '3d']
        geom_metrics = [g for g in acceptablegeometrics
                        if any(g in nm for nm in model_input_names)]

        # vegetation indices: check model input names for index keywords
        index_keywords = [
            'xyz', '3d', 'sd', 'all', 'simple',
            'exr', 'exg', 'exb', 'exgr',
            'ngrdi', 'mgrvi', 'gli', 'rgbvi', 'ikaw', 'gla',
        ]
        inferred = [kw for kw in index_keywords
                    if any(kw in nm for nm in model_input_names)]
        # deduplicate while preserving order; fall back to rgb if nothing found
        seen = set()
        indices = []
        for kw in inferred:
            if kw not in seen:
                indices.append(kw)
                seen.add(kw)
        if not indices:
            indices = ['rgb']

        feedback.pushInfo(f'Inferred indices for reclassification: {indices}')
        feedback.pushInfo(f'Geometry metrics: {geom_metrics}')

        # ---- reclassify ------------------------------------------------
        feedback.pushInfo(f'\nReclassifying {os.path.basename(reclass_file)} …')

        # predict_reclass_write() writes output next to the input file by
        # default.  We honour output_dir by temporarily changing cwd so the
        # results_<date>/ folder is created there instead.
        original_cwd = os.getcwd()
        try:
            os.chdir(output_dir)
            predict_reclass_write(
                reclass_file,
                [model],
                batch_sz=batch_size,
                ds_cache=cache,
                indiceslist=indices,
                geo_metrics=geom_metrics,
                geom_rad=geom_radius,
                write_probabilities=write_probs,
            )
        except Exception as e:
            raise QgsProcessingException(f'Reclassification failed: {e}')
        finally:
            os.chdir(original_cwd)

        feedback.pushInfo(
            f'Reclassification complete. Output written to: {output_dir}'
        )
        return {self.P_OUTPUT_DIR: output_dir}
