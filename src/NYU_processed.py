"""dataset generated from NYU-Depth V2."""

import datasets

_CITATION = """\
@inproceedings{Silberman:ECCV12,
  author    = {Nathan Silberman, Derek Hoiem, Pushmeet Kohli and Rob Fergus},
  title     = {Indoor Segmentation and Support Inference from RGBD Images},
  booktitle = {ECCV},
  year      = {2012}
}
@inproceedings{icra_2019_fastdepth,
  author    = {Wofk, Diana and Ma, Fangchang and Yang, Tien-Ju and Karaman, Sertac and Sze, Vivienne},
  title     = {FastDepth: Fast Monocular Depth Estimation on Embedded Systems},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2019}
}
"""

_DESCRIPTION = """\
The NYU-Depth V2 data set is comprised of video sequences from a variety of indoor scenes as recorded by both the RGB and Depth cameras from the Microsoft Kinect.
"""

_HOMEPAGE = "https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html"

_LICENSE = "Apace 2.0 License"

_URLS = {
    "caption_gen": "/data/NYU_processed/caption_gen.jsonl",
    "caption_ori": "/data/NYU_processed/caption_ori.jsonl",
    "data_folder": "/data/NYU_processed",
    "depth_file": "/data/NYU_processed/depth.h5py",
}

class NYUDepthV2Processed(datasets.GeneratorBasedBuilder):
    """NYU-Depth V2 dataset."""

    VERSION = datasets.Version("0.1.0")

    def _info(self):
        features = datasets.Features(
            {"image": datasets.Image(),
             "disparity": datasets.Image(),
             "depth": datasets.Array2D((512, 512), dtype="float32"),
             "text": datasets.features.Sequence(feature="string", length=4)}
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        archives = dl_manager.download(_URLS)

        # preload it here to use dataset generator in multiprocess
        import jsonlines
        with jsonlines.open(archives["caption_gen"], "r") as reader:
            metadata_gen = [r for r in reader]
        with jsonlines.open(archives["caption_ori"], "r") as reader:
            metadata_ori = [r for r in reader]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "captions": metadata_gen,
                    "data_folder": archives["data_folder"],
                    "depth_file": archives["depth_file"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "captions": metadata_ori,
                    "data_folder": archives["data_folder"],
                    "depth_file": archives["depth_file"],
                },
            ),
        ]

    def _generate_examples(self, captions, data_folder, depth_file):
        import os
        from PIL import Image
        import h5py
        # Open the HDF5 file in read-only mode
        with h5py.File(depth_file, 'r') as f:
            depth_list = f['depth']
            for idx, item in enumerate(captions):
                image = Image.open(os.path.join(data_folder, item["file_name"]))
                text = item["text"]
                img_idx = int(item["file_name"].split("/")[-1].split('.')[0].split('_')[0])
                disparity = Image.open(os.path.join(data_folder, "disparity", f"{img_idx}.png"))
                depth_map = depth_list[img_idx]
                yield idx, {"image": image, "disparity": disparity, "depth": depth_map, "text": text}