from queue import Queue
from lib.lightning_framework.callbacks import build_prediction_writer
from utils.post_processing.default_config import DEFAULT_CONFIG
from utils.post_processing.post_processing_worker import PostProcessingWorker
from utils.post_processing.post_processing_thread import PostProcessingThread, DummyBuffer, DummyPostProcessingThread
from utils.post_processing.contour_extractor import contourExtractorFactory
from utils.post_processing.statistical_analyzer import StatisticalAnalyzer
from utils.post_processing.post_processing_visualizer import Visualizer


def build_post_processor(node=DEFAULT_CONFIG):
    prediction_writer = build_prediction_writer(node.WRITER)

    contour_extractor = contourExtractorFactory(node)
    statistical_analyzer = StatisticalAnalyzer(kernel_shape=node.kernel_shape, kernel_size=node.kernel_size,
                                               multiprocessing=node.multiprocessing, **node.STATISTICAL)
    visualizer = Visualizer(multiprocessing=node.multiprocessing, **node.VISUALIZATION)
    worker = PostProcessingWorker(
        contour_extractor=contour_extractor,
        statistical_analyzer=statistical_analyzer,
        visualizer=visualizer,
        muck_num_thresh=node.muck_num_thresh,
    )

    if node.MASTER_SLAVE_MODE:
        buffer = Queue(node.BUFFER_SIZE)
        return PostProcessingThread(worker, buffer, prediction_writer, **node.EXPORT)
    else:
        buffer = DummyBuffer()
        thread = DummyPostProcessingThread(worker, buffer, prediction_writer, **node.EXPORT)
        buffer.parent = thread
        return thread

# EOF