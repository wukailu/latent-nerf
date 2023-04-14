"""dataset generated from NYU-Depth V2."""

import datasets
from glob import glob

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
    "training_metadata": "metadata.jsonl",
    "images": ["/".join(p.split("/")[-2:]) for p in glob("/data/NYU_gen/images/*.png")],
    "disparity": ["/".join(p.split("/")[-2:]) for p in glob("/data/NYU_gen/disparity/*.png")],
}


class NYUDepthV2(datasets.GeneratorBasedBuilder):
    """NYU-Depth V2 dataset."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features(
            {"image": datasets.Image(), "disparity": datasets.Image(), "text": datasets.Value("string")}
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

        file_mapping = {"/".join(p.split("/")[-2:]): p for p in archives['images']}
        file_mapping = {**file_mapping, **{"/".join(p.split("/")[-2:]): p for p in archives['disparity']}}

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metadata": archives["training_metadata"],
                    "file_mapping": file_mapping,
                },
            ),
        ]

    def _generate_examples(self, metadata, file_mapping):
        import jsonlines
        from PIL import Image
        with jsonlines.open(metadata, "r") as reader:
            for idx, item in enumerate(reader):
                image = Image.open(file_mapping[item["file_name"]]).convert("RGB")
                disparity = Image.open(file_mapping[item["disparity"]]).convert("RGB")
                text = item["text"]
                yield idx, {"image": image, "disparity": disparity, "text": text}