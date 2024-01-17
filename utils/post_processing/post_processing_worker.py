import torch


class PostProcessingWorker:
    def __init__(self, contour_extractor, statistical_analyzer, visualizer,
                 muck_num_thresh=10):
        self.contour_extractor = contour_extractor
        self.statistical_analyzer = statistical_analyzer
        self.visualizer = visualizer
        self.muck_num_thresh = muck_num_thresh

    def __call__(self, orig_image: torch.Tensor, boundary_pred: torch.Tensor, region_pred: torch.Tensor):
        contour_dict = self.contour_extractor(orig_image, boundary_pred, region_pred)

        return self.process_from_contour_dict(contour_dict, orig_image)

    def process_from_contour_dict(self, contour_dict, orig_image):
        image_size = (orig_image.shape[-2], orig_image.shape[-1])

        if len(contour_dict) < self.muck_num_thresh:
            contour_dict, statistic_dict, grain_dist_dict, grain_dist_parameters = self.statistical_analyzer.get_placeholders(
                contour_dict, image_size)
            result_imgs = {'.result': orig_image.unsqueeze(0)}
        else:
            contour_dict, statistic_dict, grain_dist_dict, grain_dist_parameters = self.statistical_analyzer(
                contour_dict, image_size)
            result_imgs = self.visualizer(orig_image, contour_dict, statistic_dict, grain_dist_dict,
                                         grain_dist_parameters)

        return result_imgs, contour_dict, statistic_dict, grain_dist_dict, grain_dist_parameters

# EOF