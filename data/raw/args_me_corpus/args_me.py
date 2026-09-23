# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""args.me Dataset"""


import json

import datasets

_CITATION = """\
@dataset{yamen_ajjour_2020_4139439,
  author       = {Yamen Ajjour and
                  Henning Wachsmuth and
                  Johannes Kiesel and
                  Martin Potthast and
                  Matthias Hagen and
                  Benno Stein},
  title        = {args.me corpus},
  month        = oct,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {1.0-cleaned},
  doi          = {10.5281/zenodo.4139439},
  url          = {https://doi.org/10.5281/zenodo.4139439}
}
"""


_DESCRIPTION = """\
The args.me corpus (version 1.0, cleaned) comprises 382 545 arguments crawled from four debate portals in the middle of 2019. The debate portals are Debatewise, IDebate.org, Debatepedia, and Debate.org. The arguments are extracted using heuristics that are designed for each debate portal.
"""

_HOMEPAGE = "https://zenodo.org/record/4139439"

_LICENSE = "https://creativecommons.org/licenses/by/4.0/legalcode"


_REPO = "https://huggingface.co/datasets/webis/args_me/resolve/main"
_URLs = {
   'corpus': f"{_REPO}/args-me.jsonl",
 #   'topics': f"{_REPO}/topics.jsonl",
 #   'judgments': f"{_REPO}/judgments.jsonl"
}


class ArgsMe(datasets.GeneratorBasedBuilder):
    """382,545 arguments crawled from debate portals"""

    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="corpus", version=VERSION, description="The args.me dataset"),
        datasets.BuilderConfig(name="topics", version=VERSION, description="The args.me dataset"),
        datasets.BuilderConfig(name="judgments", version=VERSION, description="The args.me dataset"),
    ]

    DEFAULT_CONFIG_NAME = "corpus" 

    def _info(self):
        features = datasets.Features(
            {
                "argument": datasets.Value("string"),
                "conclusion": datasets.Value("string"),
                "stance": datasets.Value("string"),
                "id": datasets.Value("string")
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        URL = _URLs[self.config.name]
        data_file = dl_manager.download(URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": data_file,
                },
            ),
        ]

    def _generate_examples(self, data_file):
        """ Yields examples as (key, example) tuples. """


        with open(data_file, encoding="utf-8") as f:
            for row in f:
                data = json.loads(row)
                id_ = data['id']
                content = data["premises"][0]
                yield id_, {
                    "argument": content['text'],
                    "conclusion": data["conclusion"],
                    "stance": content['stance'],
                    "id": id_
                }
