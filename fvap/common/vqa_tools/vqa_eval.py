"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# coding=utf-8

__author__ = "aagrawal"

# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link:
# (https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/eval.py).
import sys
import re
from .evaluate_metrics import calculate_bleu,calculate_exactmatch,calculate_f1score
from nltk.translate.bleu_score import sentence_bleu
import collections


class VQAEval:
    def __init__(self, vqa=None, vqaRes=None, n=2):
        self.n = n
        self.accuracy = {}
        self.evalQA = {}
        self.evalQuesType = {}
        self.evalAnsType = {}
        self.vqa = vqa
        self.vqaRes = vqaRes
        if vqa is not None:
            self.params = {"question_id": vqa.getQuesIds()}
        self.contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }
        self.manualMap = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        self.articles = ["a", "an", "the"]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def evaluate(self, quesIds=None):
        if quesIds == None:
            quesIds = [quesId for quesId in self.params["question_id"]]
        gts = {}
        res = {}
        for quesId in quesIds:
            gts[quesId] = self.vqa.qa[quesId]
            res[quesId] = self.vqaRes.qa[quesId]

        # =================================================
        # Compute accuracy
        # =================================================
        print("computing accuracy")
        step = 0
        num = 0
        first_key = next(iter(gts))
        if "mask_name" in gts[first_key]:
            inside_scores = collections.defaultdict(list)
            grade_scores = collections.defaultdict(list)
            whole_scores = collections.defaultdict(list)
            fovea_scores = collections.defaultdict(list)
            for quesId in quesIds:
                num += 1
                resAns = res[quesId]["answer"]
                resAns = resAns.replace("\n", " ")
                resAns = resAns.replace("\t", " ")
                resAns = resAns.strip()
                resAns = self.processPunctuation(resAns)
                resAns = self.processDigitArticle(resAns)
                gtAnswers = str(gts[quesId]["answer"])
                gtAnswers = gtAnswers.replace("\n", " ")
                gtAnswers = gtAnswers.replace("\t", " ")
                gtAnswers = gtAnswers.strip()
                gtAnswers = self.processPunctuation(gtAnswers)
                gtAnswers = self.processDigitArticle(gtAnswers)
                questionType = gts[quesId]["question_type"]
                if questionType == "inside":
                    if resAns == gtAnswers:
                        inside_scores['hit'].append(1)
                    else:
                        inside_scores['hit'].append(0)
                    inside_scores['q_id'].append(quesId)
                if questionType == "grade":
                    if resAns == gtAnswers:
                        grade_scores['hit'].append(1)
                    else:
                        grade_scores['hit'].append(0)
                    grade_scores['q_id'].append(quesId)
                if questionType == "whole":
                    if resAns == gtAnswers:
                        whole_scores['hit'].append(1)
                    else:
                        whole_scores['hit'].append(0)
                    whole_scores['q_id'].append(quesId)
                if questionType == "fovea":
                    if resAns == gtAnswers:
                        fovea_scores['hit'].append(1)
                    else:
                        fovea_scores['hit'].append(0)
                    fovea_scores['q_id'].append(quesId)
                    if step % 100 == 0:
                        self.updateProgress(step / float(len(quesIds)))
                    step = step + 1
            inside_score = sum(inside_scores['hit']) / len(inside_scores['hit'])
            grade_score = sum(grade_scores['hit']) / len(grade_scores['hit'])
            whole_score = sum(whole_scores['hit']) / len(whole_scores['hit'])
            fovea_score = sum(fovea_scores['hit']) / len(fovea_scores['hit'])
            overall = (sum(inside_scores['hit']) + sum(grade_scores['hit'])+sum(whole_scores['hit']) + sum(fovea_scores['hit'])) \
            / (len(inside_scores['hit'])+len(grade_scores['hit'])+len(whole_scores['hit'])+len(fovea_scores['hit']))
            self.accuracy["overall"] = round(overall * 100, self.n)
            self.accuracy["inside"] = round(inside_score * 100, self.n)
            self.accuracy["grade"] = round(grade_score * 100, self.n)
            self.accuracy["whole"] = round(whole_score * 100, self.n)
            self.accuracy["fovea"] = round(fovea_score * 100, self.n)
        else:
            closed_scores = collections.defaultdict(list)
            bleu_scores = collections.defaultdict(list)
            f1_scores = collections.defaultdict(list)
            open_hit_scores = collections.defaultdict(list)
            for quesId in quesIds:
                num += 1
                resAns = str(res[quesId]["answer"])
                resAns = resAns.replace("\n", " ")
                resAns = resAns.replace("\t", " ")
                resAns = resAns.strip()
                resAns = self.processPunctuation(resAns)
                resAns = self.processDigitArticle(resAns)
                gtAnswers = str(gts[quesId]["answer"])
                gtAnswers = gtAnswers.replace("\n", " ")
                gtAnswers = gtAnswers.replace("\t", " ")
                gtAnswers = gtAnswers.strip()
                gtAnswers = self.processPunctuation(gtAnswers)
                gtAnswers = self.processDigitArticle(gtAnswers)
                ansType = gts[quesId]["answer_type"]
                if ansType == "OPEN":
                    if gtAnswers in resAns:
                        open_hit_scores['hit'].append(1)
                    else:
                        open_hit_scores['hit'].append(0)
                    open_hit_scores['q_id'].append(quesId)

                    f1_score, precision, recall = calculate_f1score(resAns, gtAnswers)
                    f1_scores['f1'].append(f1_score)
                    f1_scores['precision'].append(precision)
                    f1_scores['recall'].append(recall)
                    f1_scores['q_id'].append(quesId)

                    b_score_1 = sentence_bleu(references=[str(gtAnswers).lower().split()],
                                              hypothesis=str(resAns).lower().split(), weights=(1, 0, 0, 0))
                    b_score_2 = sentence_bleu(references=[str(gtAnswers).lower().split()],
                                              hypothesis=str(resAns).lower().split(), weights=(0, 1, 0, 0))

                    bleu_scores['q_id'].append(quesId)
                    bleu_scores['bleu_score_1'].append(b_score_1)
                    bleu_scores['bleu_score_2'].append(b_score_2)

                else:
                    closed_scores['q_id'].append(quesId)
                    if 'yes' in resAns or 'no' in resAns:
                        if gtAnswers in resAns:
                            closed_scores['hit'].append(1)
                        else:
                            closed_scores['hit'].append(0)
                    elif gtAnswers in resAns:
                        closed_scores['hit'].append(1)
                    else:
                        closed_scores['hit'].append(0)

                if step % 100 == 0:
                    self.updateProgress(step / float(len(quesIds)))
                step = step + 1
            f1_score = sum(f1_scores['f1']) / len(f1_scores['f1'])
            precision = sum(f1_scores['precision']) / len(f1_scores['precision'])
            recall = sum(f1_scores['recall']) / len(f1_scores['recall'])

            bleu_score_1 = sum(bleu_scores['bleu_score_1']) / len(bleu_scores['bleu_score_1'])
            bleu_score_2 = sum(bleu_scores['bleu_score_2']) / len(bleu_scores['bleu_score_2'])

            open_score = sum(open_hit_scores['hit']) / len(open_hit_scores['hit'])
            closed_score = sum(closed_scores['hit']) / len(closed_scores['hit']) if len(closed_scores['hit']) != 0 else 0.0
            overall = (sum(open_hit_scores['hit']) + sum(closed_scores['hit'])) / (len(open_hit_scores['hit']) + len(closed_scores['hit']))
            self.accuracy["overall"] = round(overall * 100, self.n)
            self.accuracy["open"] = round(open_score * 100, self.n)
            self.accuracy["close"] = round(closed_score * 100, self.n)
            self.accuracy["f1_score"] = round(f1_score * 100, self.n)
            self.accuracy["precision"] = round(precision * 100, self.n)
            self.accuracy["recall"] = round(recall * 100, self.n)
            self.accuracy["bleu_score_1"] = round(bleu_score_1 * 100, self.n)
            self.accuracy["bleu_score_2"] = round(bleu_score_2 * 100, self.n)

        print("number of datasets:", num)
        print("Done computing accuracy")

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        return outText

    def setAccuracy(self, accQA, accQuesType, accAnsType):
        self.accuracy["overall"] = round(100 * float(sum(accQA)) / len(accQA), self.n)
        self.accuracy["perQuestionType"] = {
            quesType: round(
                100 * float(sum(accQuesType[quesType])) / len(accQuesType[quesType]),
                self.n,
            )
            for quesType in accQuesType
        }
        self.accuracy["perAnswerType"] = {
            ansType: round(
                100 * float(sum(accAnsType[ansType])) / len(accAnsType[ansType]), self.n
            )
            for ansType in accAnsType
        }

    def setEvalQA(self, quesId, acc):
        self.evalQA[quesId] = round(100 * acc, self.n)

    def setEvalQuesType(self, quesId, quesType, acc):
        if quesType not in self.evalQuesType:
            self.evalQuesType[quesType] = {}
        self.evalQuesType[quesType][quesId] = round(100 * acc, self.n)

    def setEvalAnsType(self, quesId, ansType, acc):
        if ansType not in self.evalAnsType:
            self.evalAnsType[ansType] = {}
        self.evalAnsType[ansType][quesId] = round(100 * acc, self.n)

    def updateProgress(self, progress):
        barLength = 20
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength * progress))
        text = "\rFinshed Percent: [{0}] {1}% {2}".format(
            "#" * block + "-" * (barLength - block), int(progress * 100), status
        )
        sys.stdout.write(text)
        sys.stdout.flush()
