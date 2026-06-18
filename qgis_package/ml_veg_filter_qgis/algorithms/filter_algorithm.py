# -*- coding: utf-8 -*-
"""
QGIS Processing Algorithm: Train and reclassify in a single step
=================================================================
Trains a TensorFlow MLP on labelled LAS/LAZ point clouds then
immediately applies the freshly trained model to a target point cloud,
all in one run – without saving and reloading the model from disk.

Mirrors the logic of ML_vegfilter.py (fixed).
"""
import os
import sys
import datetime

from qgis.core import (
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProcessingParameterFile,
    QgsProcessingParameterString,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFolderDestination,
    QgsProcessingOutputFolder,
    QgsProcessingException,
)

from .base import MLVegBaseAlgorithm, _ensure_src_on_path


class MLVegFilterAlgorithm(MLVegBaseAlgorithm):
    """Train a new model and immediately reclassify a point cloud."""

    P_INPUT_PCS    = 'INPUT_PCS'
    P_RECLASS_FILE = 'RECLASS_FILE'
    P_MODEL_NAME   = 'MODEL_NAME'
    P_SAVE_PLOT    = 'SAVE_PLOT'
    P_WRITE_PROBS  = 'WRITE_PROBS'
    P_OUTPUT_DIR   = 'OUTPUT_DIR'

    # ------------------------------------------------------------------
    def name(self):
        return 'train_and_reclassify'

    def displayName(self):
        return '3 – Train model and reclassify (combined)'

    def group(self):
        return 'ML Vegetation Filter'

    def groupId(self):
        return 'mlvegfilter'

    def shortHelpString(self):
        return (
            '<p><b>Train a new model and immediately reclassify a point cloud '
            'in a single step.</b></p>'
            '<p>Equivalent to running the <em>Train</em> algorithm followed by '
            'the <em>Reclassify</em> algorithm, but the model does not need to '
            'be reloaded from disk between the two steps.</p>'
            '<p><b>Training files</b> – Paste the full path to each LAS/LAZ '
            'file, one per line. Order = class label '
            '(first file → class 0, second → class 1, etc.).</p>'
            '<p><b>Point cloud to reclassify</b> – The target LAS/LAZ file to '
            'classify with the freshly trained model.</p>'
            '<p><b>Outputs</b> – The trained model files and a reclassified LAZ '
            'file are all written to the chosen output folder.</p>'
            '<p>Based on: Wernette (2024) '
            '<a href="https://doi.org/10.5281/zenodo.10966854">'
            'doi:10.5281/zenodo.10966854</a></p>'
        )

    def createInstance(self):
        return MLVegFilterAlgorithm()

    # ------------------------------------------------------------------
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterString(
                self.P_INPUT_PCS,
                'Training LAS/LAZ files – one full path per line (order = class label)',
                multiLine=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFile(
                self.P_RECLASS_FILE,
                'Point cloud to reclassify (.las or .laz)',
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.P_MODEL_NAME,
                'Output model name (no spaces)',
                defaultValue='veg_model',
            )
        )
        self._add_veg_index_params()
        self._add_model_arch_params()
        self._add_training_params()
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.P_SAVE_PLOT,
                'Save training history plot as PNG',
                defaultValue=True,
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
            QgsProcessingParameterFolderDestination(
                self.P_OUTPUT_DIR,
                'Output folder for model files and reclassified LAZ',
            )
        )
        self.addOutput(
            QgsProcessingOutputFolder(self.P_OUTPUT_DIR, 'Output folder')
        )

    # ------------------------------------------------------------------
    def processAlgorithm(self, parameters, context, feedback):
        _ensure_src_on_path()

        # ---- collect parameters ----------------------------------------
        raw_pcs     = self.parameterAsString(parameters, self.P_INPUT_PCS, context)
        filesin     = [p.strip() for p in raw_pcs.splitlines() if p.strip()]
        reclass_f   = self.parameterAsFile(parameters, self.P_RECLASS_FILE, context)
        model_name  = self.parameterAsString(parameters, self.P_MODEL_NAME, context).replace(' ', '_')
        output_dir  = self.parameterAsString(parameters, self.P_OUTPUT_DIR, context)
        save_plot   = self.parameterAsBoolean(parameters, self.P_SAVE_PLOT, context)
        write_probs = self.parameterAsBoolean(parameters, self.P_WRITE_PROBS, context)

        veg_indices, model_inputs = self._resolve_veg_indices(parameters, context)
        nodes        = self._resolve_nodes(parameters, context)
        dropout      = self.parameterAsDouble(parameters, self.P_MODEL_DROPOUT, context)
        geom_radius  = self.parameterAsDouble(parameters, self.P_GEOM_RADIUS, context)
        epochs       = self.parameterAsInt(parameters, self.P_EPOCHS, context)
        batch_size   = self.parameterAsInt(parameters, self.P_BATCH_SIZE, context)
        shuffle      = self.parameterAsBoolean(parameters, self.P_SHUFFLE, context)
        cache        = self.parameterAsBoolean(parameters, self.P_CACHE, context)
        prefetch     = self.parameterAsBoolean(parameters, self.P_PREFETCH, context)
        split        = self.parameterAsDouble(parameters, self.P_SPLIT, context)
        class_imbal  = self.parameterAsBoolean(parameters, self.P_CLASS_IMBAL, context)
        data_red     = self.parameterAsDouble(parameters, self.P_DATA_REDUCTION, context)
        es_patience  = self.parameterAsInt(parameters, self.P_EARLY_PATIENCE, context)
        es_delta     = self.parameterAsDouble(parameters, self.P_EARLY_DELTA, context)
        verbose      = self.parameterAsInt(parameters, self.P_VERBOSE, context)
        geom_metrics = self._geom_metrics_from_indices(veg_indices)

        # ---- validate --------------------------------------------------
        if len(filesin) < 2:
            raise QgsProcessingException(
                'At least two training files are required (one per class).'
            )
        for f in filesin + [reclass_f]:
            if not os.path.isfile(f):
                raise QgsProcessingException(f'File not found: {f}')
        os.makedirs(output_dir, exist_ok=True)

        # ---- imports ---------------------------------------------------
        feedback.pushInfo('Importing TensorFlow backend …')
        try:
            import tensorflow as tf
            from src.fileio import las2split, df_to_dataset, predict_reclass_write
            from src.modelbuilder import build_model
        except ImportError as e:
            raise QgsProcessingException(
                f'Required package not found: {e}\n'
                'Install with: pip install tensorflow laspy lazrs pandas '
                'scikit-learn tqdm matplotlib'
            )

        feedback.pushInfo(
            f'TensorFlow {tf.__version__}  |  '
            f'GPUs available: {len(tf.config.experimental.list_physical_devices("GPU"))}'
        )
        feedback.pushInfo(f'Model name   : {model_name}')
        feedback.pushInfo(f'Model inputs : {model_inputs}')
        feedback.pushInfo(f'Veg indices  : {veg_indices}')
        feedback.pushInfo(f'Nodes        : {nodes}')
        feedback.pushInfo(f'Geom metrics : {geom_metrics}  radius={geom_radius}m')

        # ================================================================
        # PART 1 – TRAIN
        # ================================================================
        feedback.pushInfo('\n=== PART 1: TRAINING ===')
        feedback.pushInfo('Loading and splitting training data …')
        try:
            train_ds, val_ds, eval_ds, class_dat = las2split(
                filesin,
                veg_indices=veg_indices,
                geometry_metrics=geom_metrics,
                class_imbalance_corr=class_imbal,
                training_split=split,
                data_reduction=data_red,
            )
        except Exception as e:
            raise QgsProcessingException(f'Error loading training data: {e}')

        n_classes = len(class_dat)
        feedback.pushInfo(f'Class dictionary  : {class_dat}  (n_classes={n_classes})')
        feedback.pushInfo(f'Training samples  : {len(train_ds)}')
        feedback.pushInfo(f'Validation samples: {len(val_ds)}')
        feedback.pushInfo(f'Evaluation samples: {len(eval_ds)}')

        feedback.pushInfo('Converting DataFrames to TensorFlow datasets …')

        def _to_tf(df):
            return df_to_dataset(
                df,
                targetcolname='veglab',
                label_depth=n_classes,
                shuffle=shuffle,
                cache_ds=cache,
                prefetch=prefetch,
                batch_size=batch_size,
            )

        train_tf = _to_tf(train_ds)
        val_tf   = _to_tf(val_ds)
        eval_tf  = _to_tf(eval_ds)

        feedback.pushInfo(f"Building and training model '{model_name}' …")
        try:
            mod, history, train_time = build_model(
                model_name=model_name,
                training_tf_dataset=train_tf,
                validation_tf_dataset=val_tf,
                rootdirectory=output_dir,
                nclasses=n_classes,
                nodes=nodes,
                activation_fx='relu',
                dropout_rate=dropout,
                loss_metric='mean_squared_error',
                model_optimizer='adam',
                earlystopping=[es_patience, es_delta],
                dotrain=True,
                dotrain_epochs=epochs,
                verbose=(verbose > 0),
            )
        except Exception as e:
            raise QgsProcessingException(f'Model training failed: {e}')

        feedback.pushInfo(f'Training complete in {datetime.timedelta(seconds=int(train_time))}')

        # Evaluate
        feedback.pushInfo('\nEvaluating model on held-out evaluation set …')
        model_eval  = mod.evaluate(eval_tf, verbose=verbose)
        eval_labels = ['loss', 'cross_entropy', 'cat_accuracy', 'precision', 'recall', 'auc']
        for label, val in zip(eval_labels, model_eval):
            feedback.pushInfo(f'  {label:18s}: {val:.6f}')

        # Optional training plot
        if save_plot:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle(f'{model_name} – Training History')
                ax1.plot(history.history.get('cat_accuracy', []), label='train')
                ax1.plot(history.history.get('val_cat_accuracy', []), label='val')
                ax1.set_title('Categorical Accuracy')
                ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.legend()
                ax2.plot(history.history.get('loss', []), label='train')
                ax2.plot(history.history.get('val_loss', []), label='val')
                ax2.set_title('Loss')
                ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.legend()
                plot_path = os.path.join(output_dir, f'{model_name}_PLOT_TRAINING.png')
                fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                feedback.pushInfo(f'Training plot saved: {plot_path}')
            except Exception as e:
                feedback.pushWarning(f'Could not save training plot: {e}')

        # Save model
        tf_minor = float(tf.__version__.split('.', 1)[1])
        ext = '.keras' if tf_minor >= 11.0 else '.h5'
        model_path   = os.path.join(output_dir, f'{model_name}_FULL_MODEL{ext}')
        weights_path = os.path.join(output_dir, f'{model_name}_MODEL_WEIGHTS.weights.h5')
        mod.save(model_path)
        mod.save_weights(weights_path)
        feedback.pushInfo(f'Full model saved : {model_path}')
        feedback.pushInfo(f'Weights saved    : {weights_path}')

        # Write summary
        summary_path = os.path.join(output_dir, f'{model_name}_MODEL_SUMMARY.txt')
        with open(summary_path, 'w') as fh:
            mod.summary(print_fn=lambda x: fh.write(x + '\n'))
            fh.write(f'\nInput files  : {filesin}\n')
            fh.write(f'Veg indices  : {veg_indices}\n')
            fh.write(f'Model inputs : {model_inputs}\n')
            fh.write('\nEvaluation metrics:\n')
            for label, val in zip(eval_labels, model_eval):
                fh.write(f'  {label}: {val:.6f}\n')
            fh.write(f'\nTrain time   : {datetime.timedelta(seconds=int(train_time))}\n')
            fh.write('\nClass dictionary:\n')
            for k, v in class_dat.items():
                fh.write(f'  class {v}: {k}\n')
        feedback.pushInfo(f'Summary saved    : {summary_path}')

        # ================================================================
        # PART 2 – RECLASSIFY (using the in-memory model; no reload)
        # ================================================================
        feedback.pushInfo('\n=== PART 2: RECLASSIFICATION ===')
        feedback.pushInfo(f'Reclassifying {os.path.basename(reclass_f)} …')

        # Derive geometry metrics and indices from the actual model inputs
        model_input_names = [inp.name for inp in mod.inputs]
        acceptablegeometrics = ['sd', '3d']
        reclass_geom = [g for g in acceptablegeometrics
                        if any(g in nm for nm in model_input_names)]

        index_keywords = [
            'xyz', '3d', 'sd', 'all', 'simple',
            'exr', 'exg', 'exb', 'exgr',
            'ngrdi', 'mgrvi', 'gli', 'rgbvi', 'ikaw', 'gla',
        ]
        seen = set()
        reclass_indices = []
        for kw in index_keywords:
            if kw not in seen and any(kw in nm for nm in model_input_names):
                reclass_indices.append(kw)
                seen.add(kw)
        if not reclass_indices:
            reclass_indices = ['rgb']

        feedback.pushInfo(f'Reclassification indices: {reclass_indices}')
        feedback.pushInfo(f'Reclassification geom   : {reclass_geom}')

        # Write output into output_dir
        original_cwd = os.getcwd()
        try:
            os.chdir(output_dir)
            predict_reclass_write(
                reclass_f,
                [mod],
                batch_sz=batch_size,
                ds_cache=cache,
                indiceslist=reclass_indices,
                geo_metrics=reclass_geom,
                geom_rad=geom_radius,
                write_probabilities=write_probs,
            )
        except Exception as e:
            raise QgsProcessingException(f'Reclassification failed: {e}')
        finally:
            os.chdir(original_cwd)

        feedback.pushInfo(
            f'\nAll outputs written to: {output_dir}'
        )
        return {self.P_OUTPUT_DIR: output_dir}
