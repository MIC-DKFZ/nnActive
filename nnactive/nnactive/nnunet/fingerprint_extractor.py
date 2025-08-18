import multiprocessing
import os
from time import sleep
from typing import List, Type, Union

import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import (
    isfile,
    join,
    load_json,
    maybe_mkdir_p,
    save_json,
)
from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import (
    DatasetFingerprintExtractor,
)
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import (
    determine_reader_writer_from_dataset_json,
)
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
from tqdm import tqdm


class NNActiveDatasetFingerprintExtractor(DatasetFingerprintExtractor):
    """
    Fingerprint Extractor with all functionality from nnU-Net. It saves out data to folder addTr if `use_mask_for_norm` is true in dataset.json.
    dataset.json gets rewrittten with `use_mask_for_nrom` value in convert_to_partannotated if plans would use it.

    """

    @property
    def save_addTr_folder(self) -> str:
        return "addTr"

    def analyze_case(
        self,
        image_files: List[str],
        segmentation_file: str,
        reader_writer_class: Type[BaseReaderWriter],
        num_samples: int = 10000,
    ):
        rw = reader_writer_class()
        images, properties_images = rw.read_images(image_files)
        segmentation, properties_seg = rw.read_seg(segmentation_file)

        # we no longer crop and save the cropped images before this is run. Instead we run the cropping on the fly.
        # Downside is that we need to do this twice (once here and once during preprocessing). Upside is that we don't
        # need to save the cropped data anymore. Given that cropping is not too expensive it makes sense to do it this
        # way. This is only possible because we are now using our new input/output interface.
        data_cropped, seg_cropped, bbox = crop_to_nonzero(images, segmentation)

        (
            foreground_intensities_per_channel,
            foreground_intensity_stats_per_channel,
        ) = DatasetFingerprintExtractor.collect_foreground_intensities(
            seg_cropped, data_cropped, num_samples=num_samples
        )

        # here we create a label folder filled with labels that come for free e.g. for BraTS.
        # regions where there is no brain are easy to detect and label -- we take these for nnActive datasets.
        # values that are filled with -1 are background and will be added to labelsTr with label 0.
        if self.dataset_json.get("use_mask_for_norm") is True:
            save_path = os.path.join(self.input_folder, self.save_addTr_folder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_seg = np.zeros_like(segmentation) - 1
            # print(f"Unique Seg Cropped: {np.unique(seg_cropped)}")
            save_seg[seg_cropped == -1] = 0
            # print(f"Unique Save Seg: {np.unique(save_seg)}")
            save_seg = save_seg[0]
            save_path = os.path.join(
                save_path,
                segmentation_file.split("/")[-1],
            )
            save_seg = rw.write_seg(
                save_seg,
                save_path,
                properties_seg,
            )

        spacing = properties_images["spacing"]

        shape_before_crop = images.shape[1:]
        shape_after_crop = data_cropped.shape[1:]
        relative_size_after_cropping = np.prod(shape_after_crop) / np.prod(
            shape_before_crop
        )
        return (
            shape_after_crop,
            spacing,
            foreground_intensities_per_channel,
            foreground_intensity_stats_per_channel,
            relative_size_after_cropping,
        )

    def run(self, overwrite_existing: bool = False) -> dict:
        # we do not save the properties file in self.input_folder because that folder might be read-only. We can only
        # reliably write in nnUNet_preprocessed and nnUNet_results, so nnUNet_preprocessed it is
        preprocessed_output_folder = join(nnUNet_preprocessed, self.dataset_name)
        maybe_mkdir_p(preprocessed_output_folder)
        properties_file = join(preprocessed_output_folder, "dataset_fingerprint.json")

        if not isfile(properties_file) or overwrite_existing:
            reader_writer_class = determine_reader_writer_from_dataset_json(
                self.dataset_json,
                # yikes. Rip the following line
                self.dataset[self.dataset.keys().__iter__().__next__()]["images"][0],
            )

            # determine how many foreground voxels we need to sample per training case
            num_foreground_samples_per_case = int(
                self.num_foreground_voxels_for_intensitystats // len(self.dataset)
            )

            r = []
            with multiprocessing.get_context("spawn").Pool(self.num_processes) as p:
                for k in self.dataset.keys():
                    r.append(
                        p.starmap_async(
                            self.analyze_case,
                            (
                                (
                                    self.dataset[k]["images"],
                                    self.dataset[k]["label"],
                                    reader_writer_class,
                                    num_foreground_samples_per_case,
                                ),
                            ),
                        )
                    )
                remaining = list(range(len(self.dataset)))
                # p is pretty nifti. If we kill workers they just respawn but don't do any work.
                # So we need to store the original pool of workers.
                workers = [j for j in p._pool]
                with tqdm(
                    desc=None, total=len(self.dataset), disable=self.verbose
                ) as pbar:
                    while len(remaining) > 0:
                        all_alive = all([j.is_alive() for j in workers])
                        if not all_alive:
                            raise RuntimeError(
                                "Some background worker is 6 feet under. Yuck. \n"
                                "OK jokes aside.\n"
                                "One of your background processes is missing. This could be because of "
                                "an error (look for an error message) or because it was killed "
                                "by your OS due to running out of RAM. If you don't see "
                                "an error message, out of RAM is likely the problem. In that case "
                                "reducing the number of workers might help"
                            )
                        done = [i for i in remaining if r[i].ready()]
                        for _ in done:
                            pbar.update()
                        remaining = [i for i in remaining if i not in done]
                        sleep(0.1)

            # results = ptqdm(DatasetFingerprintExtractor.analyze_case,
            #                 (training_images_per_case, training_labels_per_case),
            #                 processes=self.num_processes, zipped=True, reader_writer_class=reader_writer_class,
            #                 num_samples=num_foreground_samples_per_case, disable=self.verbose)
            results = [i.get()[0] for i in r]

            shapes_after_crop = [r[0] for r in results]
            spacings = [r[1] for r in results]
            foreground_intensities_per_channel = [
                np.concatenate([r[2][i] for r in results])
                for i in range(len(results[0][2]))
            ]
            # we drop this so that the json file is somewhat human readable
            # foreground_intensity_stats_by_case_and_modality = [r[3] for r in results]
            median_relative_size_after_cropping = np.median([r[4] for r in results], 0)

            num_channels = len(
                self.dataset_json["channel_names"].keys()
                if "channel_names" in self.dataset_json.keys()
                else self.dataset_json["modality"].keys()
            )
            intensity_statistics_per_channel = {}
            for i in range(num_channels):
                intensity_statistics_per_channel[i] = {
                    "mean": float(np.mean(foreground_intensities_per_channel[i])),
                    "median": float(np.median(foreground_intensities_per_channel[i])),
                    "std": float(np.std(foreground_intensities_per_channel[i])),
                    "min": float(np.min(foreground_intensities_per_channel[i])),
                    "max": float(np.max(foreground_intensities_per_channel[i])),
                    "percentile_99_5": float(
                        np.percentile(foreground_intensities_per_channel[i], 99.5)
                    ),
                    "percentile_00_5": float(
                        np.percentile(foreground_intensities_per_channel[i], 0.5)
                    ),
                }

            fingerprint = {
                "spacings": spacings,
                "shapes_after_crop": shapes_after_crop,
                "foreground_intensity_properties_per_channel": intensity_statistics_per_channel,
                "median_relative_size_after_cropping": median_relative_size_after_cropping,
            }

            try:
                save_json(fingerprint, properties_file)
            except Exception as e:
                if isfile(properties_file):
                    os.remove(properties_file)
                raise e
        else:
            fingerprint = load_json(properties_file)
        return fingerprint
