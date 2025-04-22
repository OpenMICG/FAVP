"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
import os

import favp.common.dist_utils as dist_utils
from favp.common.vqa_tools.vqa import SlakeVQA, DME_VQA, RADVQA
from favp.common.vqa_tools.vqa_eval import VQAEval
from favp.common.registry import registry
from favp.tasks.base_task import BaseTask


@registry.register_task("vqa")
class VQATask(BaseTask):
    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        num_ans_candidates,
        inference_method="rank",
        prompt="",
    ):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

        self.evaluate = evaluate
        self.inference_method = inference_method
        self.num_ans_candidates = num_ans_candidates
        self.prompt = prompt

        self.answer_list = None

        self.ques_files = dict()
        self.anno_files = dict()

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 3)
        max_len = run_cfg.get("max_len", 10)
        min_len = run_cfg.get("min_len", 1)

        evaluate = run_cfg.get("evaluate", False)

        inference_method = run_cfg.get("inference_method", "rank")
        num_ans_candidates = run_cfg.get("num_ans_candidates", 128)
        prompt = run_cfg.get("prompt", "")

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method,
            prompt=prompt,
        )

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)
        for dataset in datasets.values():
            for split in dataset:
                if (
                    hasattr(dataset[split], "ques_file")
                    and dataset[split].ques_file is not None
                ):

                    self.ques_files[split] = dataset[split].ques_file
                    self.anno_files[split] = dataset[split].anno_file

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."

        return datasets

    def valid_step(self, model, samples):
        answers = model.generate(
            # images=samples["image"],
            texts=samples["instruction_input"],
            image_path=samples['image_path']
        )
        pred_qa_pairs = []
        question_id = samples["question_id"]
        for answer, ques_id in zip(answers, question_id):
            ques_id = int(ques_id.item())
            pred_qa_pairs.append({"qid": ques_id, "answer": answer})
        # print("pred_qa_pairs",pred_qa_pairs)
        return pred_qa_pairs

    def after_evaluation(self, val_result, split_name, **kwargs):
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_vqa_result",
            remove_duplicate="qid",
        )
        metrics = self._report_metrics(result_file=result_file, split=split_name, datasets=kwargs['datasets'])

        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split, datasets):
        """
        Use official VQA evaluation script to report metrics.
        """
        metrics = {}
        if split in ['val', 'test']:
            datasets =str(datasets)
            if "DME" in datasets:
                vqa = DME_VQA(self.anno_files[split], self.ques_files[split])
                vqa_result = vqa.loadRes(
                    resFile=result_file, quesFile=self.ques_files[split]
                )
                vqa_scorer = VQAEval(vqa, vqa_result, n=4)
                logging.info("Start VQA evaluation.")
                vqa_scorer.evaluate()
                overall_acc = vqa_scorer.accuracy["overall"]
                metrics["agg_metrics"] = overall_acc
                logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
                logging.info("Per Question Type Accuracy is the following:")

                inside_acc = vqa_scorer.accuracy["inside"]
                metrics["inside"] = inside_acc
                logging.info("Inside Accuracy is: %.02f\n" % inside_acc)

                grade_acc = vqa_scorer.accuracy["grade"]
                metrics["grade"] = grade_acc
                logging.info("Grade Accuracy is: %.02f\n" % grade_acc)

                whole_score = vqa_scorer.accuracy['whole']
                metrics["whole"] = whole_score
                logging.info("Whole Accuracy is: %.02f\n" % whole_score)

                fovea_score = vqa_scorer.accuracy['fovea']
                metrics["fovea"] = fovea_score
                logging.info("Fovea Accuracy is: %.02f\n" % fovea_score)

            else:
                if "vqa_rad" in str(datasets):
                    vqa = RADVQA(self.anno_files[split], self.ques_files[split])
                else:
                    vqa = SlakeVQA(self.anno_files[split], self.ques_files[split])
                vqa_result = vqa.loadRes(
                    resFile=result_file, quesFile=self.ques_files[split]
                )
                # create vqaEval object by taking vqa and vqaRes
                # n is precision of accuracy (number of places after decimal), default is 2
                vqa_scorer = VQAEval(vqa, vqa_result, n=4)
                logging.info("Start VQA evaluation.")
                vqa_scorer.evaluate()

                # print accuracies
                overall_acc = vqa_scorer.accuracy["overall"]
                metrics["agg_metrics"] = overall_acc
                logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
                logging.info("Per Answer Type Accuracy is the following:")

                open_acc = vqa_scorer.accuracy["open"]
                metrics["open"] = open_acc
                logging.info("Open Accuracy is: %.02f\n" % open_acc)

                close_acc = vqa_scorer.accuracy["close"]
                metrics["close"] = close_acc
                logging.info("Close Accuracy is: %.02f\n" % close_acc)

                f1_score = vqa_scorer.accuracy['f1_score']
                metrics["f1_score"] = f1_score
                logging.info("f1_score is: %.02f\n" % f1_score)
                recall = vqa_scorer.accuracy['recall']
                metrics["recall"] = recall
                logging.info("recall is: %.02f\n" % recall)

                bleu_score_1 = vqa_scorer.accuracy['bleu_score_1']
                metrics["bleu_score_1"] = bleu_score_1
                logging.info("bleu_score_1 is: %.02f\n" % bleu_score_1)
                bleu_score_2 = vqa_scorer.accuracy['bleu_score_1']
                metrics["bleu_score_2"] = bleu_score_2
                logging.info("bleu_score_2 is: %.02f\n" % bleu_score_2)


            with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
            ) as f:
                f.write(json.dumps(metrics) + "\n")

        return metrics

