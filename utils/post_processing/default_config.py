from lib.pytorch_framework.utils import CustomCfgNode as CN
from lib.lightning_framework.callbacks import DEFAULT_CONFIG_PREDICTION_WRITER


DEFAULT_CONFIG = CN(visible=False)
DEFAULT_CONFIG.MASTER_SLAVE_MODE = False
DEFAULT_CONFIG.BUFFER_SIZE = 0 # only available for 'master_slave' mode
DEFAULT_CONFIG.multiprocessing = False
DEFAULT_CONFIG.kernel_shape = 'ellipse'
DEFAULT_CONFIG.kernel_size = 3
DEFAULT_CONFIG.WRITER = DEFAULT_CONFIG_PREDICTION_WRITER.clone()
# -----------parameters that will affect the segmentation results----------------
DEFAULT_CONFIG.EXTRACTOR = CN(visible=False)
DEFAULT_CONFIG.EXTRACTOR.PREDICTION_TYPE = 'possibility' # 'possibility' or 'binary'
DEFAULT_CONFIG.EXTRACTOR.POSSIBILITY = CN(visible=False)
DEFAULT_CONFIG.EXTRACTOR.POSSIBILITY.region_prob_thresh = 0.4
DEFAULT_CONFIG.EXTRACTOR.POSSIBILITY.boundary_prob_shift = 0.1
DEFAULT_CONFIG.EXTRACTOR.POSSIBILITY.center_prob_thresh = 0.7
DEFAULT_CONFIG.EXTRACTOR.POSSIBILITY.center_open_iter = 1
DEFAULT_CONFIG.EXTRACTOR.POSSIBILITY.erosion_mode = 'direct'
DEFAULT_CONFIG.EXTRACTOR.POSSIBILITY.max_erosion_iter = 10
DEFAULT_CONFIG.EXTRACTOR.POSSIBILITY.center_rel_dist_thresh = 3.0
DEFAULT_CONFIG.EXTRACTOR.POSSIBILITY.gaussian_kernel_size = 15
DEFAULT_CONFIG.EXTRACTOR.POSSIBILITY.apply_stage2 = True
DEFAULT_CONFIG.EXTRACTOR.BINARY = CN(visible=False)
DEFAULT_CONFIG.EXTRACTOR.BINARY.erosion_lambda = 0.15
# -----------parameters that will affect the geometrical characteristic calculation----------------
DEFAULT_CONFIG.STATISTICAL = CN(visible=False)
DEFAULT_CONFIG.STATISTICAL.pixel_ratio = 0.44
DEFAULT_CONFIG.STATISTICAL.redilate_iter = 2
# -----------parameters that will affect the visualization appearance----------------
DEFAULT_CONFIG.VISUALIZATION = CN(visible=False)
DEFAULT_CONFIG.VISUALIZATION.visualization_mode = 'log_area'
DEFAULT_CONFIG.VISUALIZATION.visualization_alpha = 0.5
DEFAULT_CONFIG.VISUALIZATION.colormap = 'turbo'
DEFAULT_CONFIG.VISUALIZATION.draw_contours = True
DEFAULT_CONFIG.VISUALIZATION.draw_legend = True
DEFAULT_CONFIG.VISUALIZATION.draw_graindist = True
DEFAULT_CONFIG.VISUALIZATION.draw_statistics = True
DEFAULT_CONFIG.VISUALIZATION.independent_graphs = False
DEFAULT_CONFIG.VISUALIZATION.landscape_mode = False
DEFAULT_CONFIG.VISUALIZATION.additional_mpl_rc = CN()
DEFAULT_CONFIG.VISUALIZATION.result_format = 'png'
DEFAULT_CONFIG.VISUALIZATION.legend_and_graphs = CN(visible=False)
DEFAULT_CONFIG.VISUALIZATION.legend_and_graphs.GLOBAL_ORIGIN = (0, 0)
DEFAULT_CONFIG.VISUALIZATION.legend_and_graphs.LEGEND_SCALE = 3
DEFAULT_CONFIG.VISUALIZATION.legend_and_graphs.LEGEND_WIDTH = 250
DEFAULT_CONFIG.VISUALIZATION.legend_and_graphs.LEGEND_ANCHOR = (30, 30)
DEFAULT_CONFIG.VISUALIZATION.legend_and_graphs.GRAPH_SIZE = (800, 320)
DEFAULT_CONFIG.VISUALIZATION.legend_and_graphs.GRAPH_SPACING = 100
DEFAULT_CONFIG.VISUALIZATION.legend_and_graphs.GRAIN_DIST_GRAPH_WIDTH = 350
# -----------threshold for applying post processing----------------
DEFAULT_CONFIG.muck_num_thresh = 10
# -----------grain size groups for the grain distribution calculation-------------
DEFAULT_CONFIG.STATISTICAL.grain_size_group = [5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300]
# -----------parameters that will affect the exported annotations----------------
DEFAULT_CONFIG.EXPORT = CN(visible=False)
DEFAULT_CONFIG.EXPORT.export_annotations = False
DEFAULT_CONFIG.EXPORT.export_simplified = True
DEFAULT_CONFIG.EXPORT.export_simplify_eps = 1


