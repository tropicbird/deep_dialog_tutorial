import sentencepiece as spm
import numpy as np
import random
from typing import List, Sequence, Tuple

ENCODER_INPUT_NODE = 'transformer/encoder_input:0'
DECODER_INPUT_NODE = 'transformer/decoder_input:0'
IS_TRAINING_NODE = 'transformer/is_training:0'


class BatchGenerator:
    def __init__(
            self,
            max_length=50,
            #夏目漱石の青空文庫で作成したSentencePieceモデルは既存のものを使用。
            #SentencePieceモデルの作り方はこれ-> https://www.pytry3g.com/entry/how-to-use-sentencepiece
            spm_model_path: str = 'deepdialog/transformer/preprocess/spm_natsume.model'
    ) -> None:
        self.max_length = max_length
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_model_path)
        self.bos = self.sp.piece_to_id('<s>')
        self.eos = self.sp.piece_to_id('</s>')
        self.pad = 0

    @property
    def vocab_size(self) -> int:
        #語彙数
        return self.sp.get_piece_size()

    def load(self, file_path: str) -> None:
        with open(file_path, encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
            '''
            lines<-
            ['虚子に誘われて珍らしく明治座を見に行った。',
             '芝居というものには全く無知無識であるから、どんな印象を受けるか自分にもまるで分らなかった。',...]  
            '''
        self.data = self._create_data(lines) #self.data: list(zip(questions, answers))

    def get_batch(self, batch_size: int = 128, shuffle=True):
        while True:
            if shuffle:
                random.shuffle(self.data)#リストをシャッフル
            raw_batch_list = self._split(self.data, batch_size) #[[[q1,a11],[q2,a2],...,[q64,a64]],[[q1,a1],[q2,a2],...,[q,a]],...]
            for raw_batch in raw_batch_list:
                questions, answers = zip(*raw_batch)
                yield {
                    ENCODER_INPUT_NODE: self._convert_to_array(questions),
                    DECODER_INPUT_NODE: self._convert_to_array(answers),
                    IS_TRAINING_NODE: True,
                }

    def _create_data(self, lines: Sequence[str]) -> List[Tuple[List[int], List[int]]]:
        questions = [self._create_question(line) for line in lines[:-1]]
        answers = [self._create_answer(line) for line in lines[1:]]
        '''
        questionsの行をt行目とした時、answerはt+1行目
        '''
        return list(zip(questions, answers))

    def _create_question(self, sentence) -> List[int]:
        ids = self.sp.encode_as_ids(sentence)
        return ids[:self.max_length] #self.max_length:50

    def _create_answer(self, sentence: str) -> List[int]:
        ids = self.sp.encode_as_ids(sentence)
        # [self.bos]:1, [self.eos]:2
        return [self.bos] + ids[:self.max_length - 2] + [self.eos]

    def _split(self, nd_list: Sequence, batch_size: int) -> List[List]:
        return [list(nd_list[i - batch_size:i]) for i in range(batch_size, len(nd_list) + 1, batch_size)]

    def _convert_to_array(self, id_list_list: Sequence[Sequence[int]]) -> np.ndarray:
        max_len = max([len(id_list) for id_list in id_list_list]) #最大長のtokenを確認

        return np.array(
            [list(id_list) + [self.pad] * (max_len - len(id_list)) for id_list in id_list_list],
            dtype=np.int32,
        )#最大長に満たない場合はself.pad=0で埋める。
