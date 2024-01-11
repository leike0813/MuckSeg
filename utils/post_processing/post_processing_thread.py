import threading
from pathlib import Path
from PIL import Image as PILImage
from torchvision.transforms.functional import to_tensor
from .post_processing_utils import write_statistics, export_annotations, import_annotations


class PostProcessingThread(threading.Thread):
    def __init__(self, worker, buffer, prediction_writer, export_annotations=True, export_simplified=True, export_simplify_eps=1):
        self.worker = worker
        self.buffer = buffer
        self.prediction_writer = prediction_writer
        self.export_annotations = export_annotations
        self.export_simplified = export_simplified
        self.export_simplify_eps = export_simplify_eps
        self._stop_event = threading.Event()
        self.results = {}
        super().__init__()

    def process(self):
        orig_image, boundary, region, img_name = self.buffer.get()
        result_imgs, contour_dict, statistic_dict, grain_dist_dict, grain_dist_parameters = self.worker(orig_image, boundary, region)
        img_paths = self.prediction_writer.write_prediction(
            result_imgs, [img_name], self.prediction_writer.output_dir, False,
            False, self.prediction_writer.log_folder, None, self.prediction_writer.image_format
        )
        return img_name, img_paths, (contour_dict, statistic_dict, grain_dist_dict, grain_dist_parameters)

    def run(self):
        result_statistics = {}
        image_paths = []
        while not self.stopped():
            img_name, img_paths, muck_statistics = self.process()
            result_statistics[img_name] = muck_statistics
            image_paths.extend(img_paths)
        while not self.buffer.empty():
            # print('Remains: {} images unprocessed'.format(self.buffer.qsize()))
            img_name, img_paths, muck_statistics = self.process()
            result_statistics[img_name] = muck_statistics
            image_paths.extend(img_paths)

        statistics_file = write_statistics(self.prediction_writer.output_dir, result_statistics)
        if self.export_annotations:
            annotations_file = export_annotations(self.prediction_writer.output_dir, result_statistics,
                                                  simplify=self.export_simplified, simplify_eps=self.export_simplify_eps)

        self.results['image_paths'] = image_paths
        self.results['statistics_file_path'] = statistics_file

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class DummyBuffer:
    def __init__(self, parent=None):
        self.parent = parent

    def put(self, item):
        self.parent.buffer_proxy(item)

    def full(self):
        return False


class DummyPostProcessingThread:
    def __init__(self, worker, buffer, prediction_writer, export_annotations=True, export_simplified=True, export_simplify_eps=1):
        self.worker = worker
        self.buffer = buffer
        self.prediction_writer = prediction_writer
        self.export_annotations = export_annotations
        self.export_simplified = export_simplified
        self.export_simplify_eps = export_simplify_eps
        self.results = {'image_paths': []}
        self.result_statistics = {}
        self._is_stopped = True

    def buffer_proxy(self, item):
        orig_image, boundary, region, img_name = item
        result_imgs, contour_dict, statistic_dict, grain_dist_dict, grain_dist_parameters = self.worker(orig_image, boundary, region)
        img_paths = self.prediction_writer.write_prediction(
            result_imgs, [img_name], self.prediction_writer.output_dir, False,
            False, self.prediction_writer.log_folder, None, self.prediction_writer.image_format
        )
        self.results['image_paths'].extend(img_paths)
        self.result_statistics[img_name] = (contour_dict, statistic_dict, grain_dist_dict, grain_dist_parameters)

    def join(self):
        statistics_file = write_statistics(self.prediction_writer.output_dir, self.result_statistics)
        if self.export_annotations:
            annotations_file = export_annotations(self.prediction_writer.output_dir, self.result_statistics,
                                                  simplify=self.export_simplified, simplify_eps=self.export_simplify_eps)
        self.results['statistics_file_path'] = statistics_file

    def stop(self):
        self._is_stopped = True

    def stopped(self):
        return self._is_stopped

    def setDaemon(self, placeholder):
        pass

    def start(self):
        self._is_stopped = False

    def process_annotations(self, annotation_path, orig_img_folder=None):
        output = import_annotations(annotation_path, orig_img_folder=orig_img_folder)
        for img_name, v in output.items():
            orig_image = v['orig_image']
            contour_dict = v['contour_dict']
            result_imgs, contour_dict, statistic_dict, grain_dist_dict, grain_dist_parameters = self.worker.process_from_contour_dict(contour_dict, orig_image)
            self.results['image_paths'].extend(self.prediction_writer.write_prediction(
                result_imgs, [Path(img_name).stem], self.prediction_writer.output_dir, False,
                False, self.prediction_writer.log_folder, None, self.prediction_writer.image_format
            ))
            self.result_statistics[Path(img_name).stem] = (contour_dict, statistic_dict, grain_dist_dict, grain_dist_parameters)

        statistics_file = write_statistics(self.prediction_writer.output_dir, self.result_statistics)
        self.results['statistics_file_path'] = statistics_file

    def process_labels(self, orig_img_folder, label_folder):
        orig_img_folder = Path(orig_img_folder)
        label_folder = Path(label_folder)
        assert orig_img_folder.is_dir(), 'Invalid original image folder path'
        assert label_folder.is_dir(), 'Invalid label folder path'

        for img_path in orig_img_folder.glob('*.jpg'):
            img_name = img_path.stem
            boundary_path = label_folder / '{}_boundary.png'.format(img_name)
            region_path = label_folder / '{}_region.png'.format(img_name)
            if boundary_path.is_file() and region_path.is_file():
                orig_image = to_tensor(PILImage.open(img_path).convert('L'))
                boundary = to_tensor(PILImage.open(boundary_path).convert('L'))
                region = to_tensor(PILImage.open(region_path).convert('L'))
                result_imgs, contour_dict, statistic_dict, grain_dist_dict, grain_dist_parameters = self.worker(
                    orig_image, boundary, region)
                img_paths = self.prediction_writer.write_prediction(
                    result_imgs, [img_name], self.prediction_writer.output_dir, False,
                    False, self.prediction_writer.log_folder, None, self.prediction_writer.image_format
                )
                self.results['image_paths'].extend(img_paths)
                self.result_statistics[img_name] = (contour_dict, statistic_dict, grain_dist_dict, grain_dist_parameters)

        self.join()

# EOF