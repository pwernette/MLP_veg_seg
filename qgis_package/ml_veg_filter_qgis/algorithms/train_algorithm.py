# -*- coding: utf-8 -*-
"""
QGIS Processing Algorithm: Train ML Vegetation Filter Model
============================================================
Trains a TensorFlow MLP on two or more labelled LAS/LAZ point clouds
(one file per class) and saves the model + metadata to a chosen folder.

Mirrors the logic of ML_veg_train.py.
"""
import os
import sys
import datetime

from qgis.core import (
    QgsProcessingContext,
    QgsProcessingFeedback,
    QgsProcessingParameterString,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFolderDestination,
    QgsProcessingOutputFolder,
    QgsProcessingException,
)

from .base import MLVegBaseAlgorithm, _ensure_src_on_path


class MLVegTrainAlgorithm(MLVegBaseAlgorithm):
    """Train a new MLP model from two or more labelled point clouds."""

    P_INPUT_PCS  = 'INPUT_PCS'
    P_MODEL_NAME = 'MODEL_NAME'
    P_SAVE_PLOT  = 'SAVE_PLOT'
    P_OUTPUT_DIR = 'OUTPUT_DIR'

    # ------------------------------------------------------------------
    def name(self):
        return 'train_model'

    def displayName(self):
        return '1 – Train vegetation filter model'

    def group(self):
        return 'ML Vegetation Filter'

    def groupId(self):
        return 'mlvegfilter'

    def shortHelpString(self):
        return (
            '<p><b>Train a TensorFlow MLP to classify vegetation vs bare-Earth '
            'in RGB-coloured LAS/LAZ point clouds.</b></p>'
            '<p><b>Training files</b> – Paste the full path to each LAS/LAZ file, '
            'one per line. The <em>order determines the class label</em>: '
            'first file → class 0, second file → class 1, and so on. '
            'Typically class 0 = bare Earth, class 1 = vegetation.</p>'
            '<p><b>Vegetation index preset</b> – Controls which spectral features '
            'are fed to the model. "rgb" is the fastest option; "all" gives the '
            'most information but is slower.</p>'
            '<p><b>Outputs</b> saved to the chosen folder:<br>'
            '&nbsp;• <code>&lt;name&gt;_FULL_MODEL.keras</code> – full trained model<br>'
            '&nbsp;• <code>&lt;name&gt;_BEST.keras</code> – best checkpoint<br>'
            '&nbsp;• <code>&lt;name&gt;_MODEL_WEIGHTS.h5</code> – weights only<br>'
            '&nbsp;• <code>&lt;name&gt;_MODEL_SUMMARY.txt</code> – architecture + metrics<br>'
            '&nbsp;• <code>&lt;name&gt;_LOG.csv</code> – per-epoch training log<br>'
            '&nbsp;• <code>&lt;name&gt;_PLOT_TRAINING.png</code> – accuracy/loss plot (optional)</p>'
            '<p>Based on: Wernette (2024) '
            '<a href="https://doi.org/10.5281/zenodo.10966854">'
            'doi:10.5281/zenodo.10966854</a></p>'
        )

    def createInstance(self):
        return MLVegTrainAlgorithm()

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
            QgsProcessingParameterFolderDestination(
                self.P_OUTPUT_DIR,
                'Output folder for saved model and logs',
            )
        )
        self.addOutput(
            QgsProcessingOutputFolder(self.P_OUTPUT_DIR, 'Saved model folder')
        )

    # ------------------------------------------------------------------
    def processAlgorithm(self, parameters, context, feedback):
        _ensure_src_on_path()

        # ---- collect parameters ----------------------------------------
        raw_pcs    = self.parameterAsString(parameters, self.P_INPUT_PCS, context)
        filesin    = [p.strip() for p in raw_pcs.splitlines() if p.strip()]
        model_name = self.parameterAsString(parameters, self.P_MODEL_NAME, context).replace(' ', '_')
        output_dir = self.parameterAsString(parameters, self.P_OUTPUT_DIR, context)
        save_plot  = self.parameterAsBoolean(parameters, self.P_SAVE_PLOT, context)

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

        # ---- validate inputs -------------------------------------------
        if len(filesin) < 2:
            raise QgsProcessingException(
                'At least two LAS/LAZ files are required (one per class).'
            )
        for f in filesin:
            if not os.path.isfile(f):
                raise QgsProcessingException(f'Input file not found: {f}')
        os.makedirs(output_dir, exist_ok=True)

        # ---- import backend --------------------------------------------
        feedback.pushInfo('Importing TensorFlow backend …')
        try:
            import tensorflow as tf
            from src.fileio import las2split, df_to_dataset
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
        feedback.pushInfo(f'Dropout      : {dropout}')
        feedback.pushInfo(f'Geom metrics : {geom_metrics}  radius={geom_radius}m')

        # ---- load and split data ---------------------------------------
        feedback.pushInfo('\nLoading and splitting training data …')
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
            raise QgsProcessingException(f'Error loading point clouds: {e}')

        n_classes = len(class_dat)
        feedback.pushInfo(f'\nClass dictionary : {class_dat}  (n_classes={n_classes})')
        feedback.pushInfo(f'Training samples : {len(train_ds)}')
        feedback.pushInfo(f'Validation samples: {len(val_ds)}')
        feedback.pushInfo(f'Evaluation samples: {len(eval_ds)}')

        # ---- convert to TF datasets ------------------------------------
        feedback.pushInfo('\nConverting DataFrames to TensorFlow datasets …')

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

        # ---- build and train -------------------------------------------
        feedback.pushInfo(f"\nBuilding and training model '{model_name}' …")
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

        # ---- evaluate --------------------------------------------------
        feedback.pushInfo('\nEvaluating model on held-out evaluation set …')
        model_eval  = mod.evaluate(eval_tf, verbose=verbose)
        eval_labels = ['loss', 'cross_entropy', 'cat_accuracy', 'precision', 'recall', 'auc']
        for label, val in zip(eval_labels, model_eval):
            feedback.pushInfo(f'  {label:18s}: {val:.6f}')

        # ---- optional training plot ------------------------------------
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
                ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
                ax1.legend()

                ax2.plot(history.history.get('loss', []), label='train')
                ax2.plot(history.history.get('val_loss', []), label='val')
                ax2.set_title('Loss')
                ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
                ax2.legend()

                plot_path = os.path.join(output_dir, f'{model_name}_PLOT_TRAINING.png')
                fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                feedback.pushInfo(f'Training plot saved : {plot_path}')
            except Exception as e:
                feedback.pushWarning(f'Could not save training plot: {e}')

        # ---- save model ------------------------------------------------
        tf_minor = float(tf.__version__.split('.', 1)[1])
        ext = '.keras' if tf_minor >= 11.0 else '.h5'
        model_path   = os.path.join(output_dir, f'{model_name}_FULL_MODEL{ext}')
        weights_path = os.path.join(output_dir, f'{model_name}_MODEL_WEIGHTS.h5')
        mod.save(model_path)
        mod.save_weights(weights_path)
        feedback.pushInfo(f'\nFull model saved    : {model_path}')
        feedback.pushInfo(f'Weights saved       : {weights_path}')

        # ---- write summary text file -----------------------------------
        summary_path = os.path.join(output_dir, f'{model_name}_MODEL_SUMMARY.txt')
        with open(summary_path, 'w') as fh:
            mod.summary(print_fn=lambda x: fh.write(x + '\n'))
            fh.write(f'\nInput files   : {filesin}\n')
            fh.write(f'Veg indices   : {veg_indices}\n')
            fh.write(f'Model inputs  : {model_inputs}\n')
            fh.write('\nEvaluation metrics:\n')
            for label, val in zip(eval_labels, model_eval):
                fh.write(f'  {label}: {val:.6f}\n')
            fh.write(f'\nTrain time    : {datetime.timedelta(seconds=int(train_time))}\n')
            fh.write('\nClass dictionary:\n')
            for k, v in class_dat.items():
                fh.write(f'  class {v}: {k}\n')
        feedback.pushInfo(f'Summary saved       : {summary_path}')

        return {self.P_OUTPUT_DIR: output_dir}
