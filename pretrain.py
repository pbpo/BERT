# pretrain_ultimate.py
# 모든 코드(데이터 로더, 파서, 토크나이저, 데이터셋, 모델, 훈련 로직)를 융합하고,
# 멀티코어 병렬 처리(공유 메모리 포함) 및 JIT 컴파일로 성능을 극한까지 최적화한 최종 버전

import argparse
import os
import json
import torch
import torch.nn as nn
import pandas as pd
import random
import re
from itertools import chain
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel, get_linear_schedule_with_warmup
from transformers.models.bert.modeling_bert import BertLMPredictionHead
import multiprocessing as mp
from functools import partial
import ctypes
from typing import List, Dict, Optional

# ==============================================================================
# 1. 핵심 구성 요소 정의 (Tokenizer, Model)
# ==============================================================================

class CANTokenizer:
    """CAN 데이터의 어휘집을 관리하고 토큰-인덱스 변환을 처리합니다."""
    def __init__(self):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.ID_OFFSET = 260
        self.special_tokens = ['<PAD>', '<UNK>', '<MASK>', '<VOID>']
        for token in self.special_tokens:
            self._add_token(token)

    def _add_token(self, token: str):
        if token not in self.token_to_id:
            index = len(self.token_to_id)
            self.token_to_id[token] = index
            self.id_to_token[index] = token

    def build_vocab(self, df: pd.DataFrame) -> None:
        """오프셋 기반 통합 어휘집을 구축합니다."""
        for i in range(256):
            self._add_token(f'{i:02X}')
        # CAN ID가 모두 문자열이라고 가정하고 처리
        for can_id in df['CAN_ID'].astype(str).unique():
            try:
                # 16진수 문자열을 정수로 변환 시도
                token = str(int(can_id, 16) + self.ID_OFFSET)
                self._add_token(token)
            except ValueError:
                # 변환에 실패하면 (예: 비정상적인 ID 형식) 건너뜁니다.
                print(f"Warning: Could not parse CAN ID '{can_id}' as hex. Skipping.")
                continue

    def load_vocab(self, file_path: str) -> None:
        """JSON 파일에서 어휘집을 로드합니다."""
        with open(file_path, 'r', encoding='utf-8') as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode(self, tokens: list) -> list[int]:
        """토큰 리스트를 정수 ID 리스트로 변환합니다."""
        unk_id = self.token_to_id['<UNK>']
        return [self.token_to_id.get(str(token), unk_id) for token in tokens]

class CANBertForMaskedLM(nn.Module):
    """CAN-BERT 모델 아키텍처."""
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        self.cls = BertLMPredictionHead(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        
        output = (prediction_scores,)
        return (masked_lm_loss,) + output if masked_lm_loss is not None else output

# ==============================================================================
# 2. 멀티코어 병렬 처리가 적용된 초고속 데이터 로더 및 데이터셋
# ==============================================================================
class HybridStreamingMLMDataset(Dataset):
    """하이브리드 스트리밍 방식의 CAN MLM 데이터셋."""
    def __init__(self, file_path: str, tokenizer: CANTokenizer, seq_len: int, mask_prob: float = 0.15, max_queue_size=1024):
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.mask_token_id = tokenizer.token_to_id['<MASK>']
        self.vocab_size = len(tokenizer.token_to_id)
        self.special_token_ids = {tokenizer.token_to_id[token] for token in tokenizer.special_tokens}

        # 데이터 전체를 로드하지 않고, 메모리 절약형 토큰 스트림을 파일로부터 생성
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

        print(f"Dataset: Loaded {len(self.lines):,} lines. Sequences will be generated on-the-fly.")
        self.token_stream = self._build_token_stream()

    def _build_token_stream(self) -> List[int]:
        token_stream = []
        for line in self.lines:
            parsed = _parse_candump_line(line)
            if parsed:
                can_id_token = str(int(parsed['CAN_ID'], 16) + self.tokenizer.ID_OFFSET)
                frame_tokens = [can_id_token] + parsed['Data']
                token_stream.extend(self.tokenizer.encode(frame_tokens))
        return token_stream

    def __len__(self) -> int:
        return max(0, len(self.token_stream) - self.seq_len)

    def __getitem__(self, idx: int) -> dict:
        input_ids = self.token_stream[idx : idx + self.seq_len]
        labels = [-100] * self.seq_len

        candidate_indices = [i for i, token_id in enumerate(input_ids) if token_id not in self.special_token_ids]
        num_to_mask = int(len(candidate_indices) * self.mask_prob)

        if num_to_mask > 0:
            masked_indices = random.sample(candidate_indices, num_to_mask)
            for i in masked_indices:
                labels[i] = input_ids[i]
                rand = random.random()
                if rand < 0.8:
                    input_ids[i] = self.mask_token_id
                elif rand < 0.9:
                    input_ids[i] = random.randint(len(self.special_token_ids), self.vocab_size - 1)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.ones(self.seq_len, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# --- 병렬 처리를 위한 글로벌 변수 및 헬퍼 함수 정의 ---
shared_token_stream = None
g_tokenizer = None

def _init_worker(shared_array_base, tokenizer_obj):
    """각 워커 프로세스 초기화 시 공유 메모리와 토크나이저를 전역 변수에 할당"""
    global shared_token_stream, g_tokenizer
    if shared_array_base:
        shared_token_stream = shared_array_base
    if tokenizer_obj:
        g_tokenizer = tokenizer_obj

def _parse_candump_line(line: str) -> Optional[Dict]:
    """[병렬 처리용 헬퍼 함수] 'candump' 형식의 로그 한 줄을 파싱합니다."""
    match = re.match(r'\s*\(\d+\.\d+\)\s+\w+\s+([0-9A-F]+)#([0-9A-F]*)', line)
    if not match: return None
    can_id, payload_str = match.groups()
    data = [payload_str[i:i+2] for i in range(0, len(payload_str), 2)]
    return {'CAN_ID': can_id, 'Data': data}

def _parse_and_tokenize_chunk(lines: List[str]) -> List[int]:
    """[병렬 처리용] 데이터 청크를 파싱하고 토큰 ID로 변환"""
    global g_tokenizer
    token_stream = []
    for line in lines:
        parsed = _parse_candump_line(line)
        if parsed:
            can_id_token = str(int(parsed['CAN_ID'], 16) + g_tokenizer.ID_OFFSET)
            frame_tokens = [can_id_token] + parsed['Data']
            token_stream.extend(g_tokenizer.encode(frame_tokens))
    return token_stream

def _create_sequences_from_shared_memory(indices: tuple) -> list:
    """[병렬 처리용] 공유 메모리에서 직접 데이터를 읽어 시퀀스 생성"""
    global shared_token_stream
    start_idx, end_idx, seq_len = indices
    # ctypes 배열을 numpy 배열처럼 슬라이싱하여 사용
    return [shared_token_stream[i : i + seq_len] for i in range(start_idx, end_idx - seq_len + 1)]

def _parallel_load_can_data(file_path: str, num_cores: int) -> pd.DataFrame:
    """병렬 처리로 CAN 데이터를 로드하고 DataFrame으로 변환합니다."""
    print(f"Loading CAN data from {file_path} using {num_cores} cores...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    chunk_size = (len(lines) + num_cores - 1) // num_cores
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    
    def _parse_chunk_to_dataframe(lines_chunk: List[str]) -> List[Dict]:
        """청크를 파싱하여 딕셔너리 리스트로 변환"""
        parsed_data = []
        for line in lines_chunk:
            parsed = _parse_candump_line(line)
            if parsed:
                parsed_data.append(parsed)
        return parsed_data
    
    with mp.Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(_parse_chunk_to_dataframe, chunks), 
                           total=len(chunks), desc="Parallel Data Loading"))
    
    # 결과를 하나의 리스트로 병합
    all_data = list(chain.from_iterable(results))
    
    # DataFrame으로 변환
    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df)} CAN frames for vocabulary building")
    
    return df

class UltimateMLMDataset(Dataset):
    """공유 메모리 병렬 처리를 통해 데이터 전처리 속도를 극한으로 끌어올린 데이터셋."""
    def __init__(self, file_path: str, tokenizer: CANTokenizer, seq_len: int, mask_prob: float = 0.15):
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        self.tokenizer = tokenizer
        
        num_cores = 20
        print(f"Dataset: Parsing and tokenizing in parallel using {num_cores} cores...")
        with open(file_path, 'r', encoding='utf-8') as f: lines = f.readlines()
        
        chunk_size = (len(lines) + num_cores - 1) // num_cores
        chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
        
        with mp.Pool(initializer=_init_worker, initargs=(None, tokenizer)) as pool:
            results = list(tqdm(pool.imap(_parse_and_tokenize_chunk, chunks), total=len(chunks), desc="Parallel Parsing & Tokenizing"))
        
        token_id_stream = list(chain.from_iterable(results))
        print(f"Dataset: Token stream length: {len(token_id_stream):,}")

        print("Dataset: Loading token stream into shared memory...")
        shared_array = mp.Array(ctypes.c_long, token_id_stream, lock=False)
        
        print(f"Dataset: Pre-generating sequences from shared memory in parallel...")
        chunk_size = (len(shared_array) + num_cores - 1) // num_cores
        tasks = [
            (i * chunk_size, min((i + 1) * chunk_size, len(shared_array)), seq_len)
            for i in range(num_cores)
        ]
        
        with mp.Pool(initializer=_init_worker, initargs=(shared_array, None)) as pool:
            results = list(tqdm(pool.imap(_create_sequences_from_shared_memory, tasks), total=len(tasks), desc="Parallel Sequence Generation"))
        
        self.samples = list(chain.from_iterable(results))
        
        self.mask_token_id = tokenizer.token_to_id['<MASK>']
        self.vocab_size = len(tokenizer.token_to_id)
        self.special_token_ids = {tokenizer.token_to_id[token] for token in tokenizer.special_tokens}

        if not self.samples: raise ValueError("No sequences were generated.")
        print(f"Dataset: Initialization complete. Total sequences in memory: {len(self.samples):,}")

    def __len__(self) -> int: return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sequence = self.samples[idx]
        input_ids = list(sequence)
        labels = [-100] * self.seq_len
        candidate_indices = [i for i, token_id in enumerate(input_ids) if token_id not in self.special_token_ids]
        num_to_mask = int(len(candidate_indices) * self.mask_prob)
        if num_to_mask > 0:
            masked_indices = random.sample(candidate_indices, num_to_mask)
            for i in masked_indices:
                labels[i] = input_ids[i]
                rand = random.random()
                if rand < 0.8: input_ids[i] = self.mask_token_id
                elif rand < 0.9: input_ids[i] = random.randint(len(self.special_token_ids), self.vocab_size - 1)
        return {'input_ids': torch.tensor(input_ids), 'attention_mask': torch.ones(self.seq_len, dtype=torch.long), 'labels': torch.tensor(labels)}

# ==============================================================================
# 3. 훈련 파이프라인 함수 정의
# ==============================================================================
# ==============================================================================
# 3. 훈련 파이프라인 함수 정의
# ==============================================================================
def train_one_epoch(model, data_loader, optimizer, scheduler, device, epoch, epochs, use_amp):
    model.train()
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
    # The GradScaler should be initialized outside the loop for each epoch
    scaler = torch.amp.GradScaler(enabled=use_amp)
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        #
        # === FIX IS HERE ===
        # Pass the device type to autocast
        #
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)[0]
        
        # Scale the loss and backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        progress_bar.set_postfix({'loss': loss.item()})

def save_checkpoint(model, optimizer, scheduler, epoch, output_dir):
    """학습 체크포인트를 저장합니다."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    checkpoint_path = os.path.join(output_dir, f"can-bert-ultimate-epoch-{epoch+1}.pt")
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)
    print(f"\nEpoch {epoch+1} | Checkpoint saved to {checkpoint_path}")

# ==============================================================================
# 4. 메인 함수
# ==============================================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print(f"Using device: {device} | Using AMP: {use_amp}")
    
    tokenizer = CANTokenizer()
    if not os.path.exists(args.vocab_path):
        print("Vocabulary not found. Building from data file...")
        df_for_vocab = _parallel_load_can_data(args.data_path, mp.cpu_count())
        tokenizer.build_vocab(df_for_vocab)
        with open(args.vocab_path, 'w') as f: json.dump(tokenizer.token_to_id, f)
        print(f"Vocabulary built and saved to {args.vocab_path}")
    else:
        tokenizer.load_vocab(args.vocab_path)

    config = BertConfig(
        vocab_size=len(tokenizer.token_to_id),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.seq_len
    )
    model = CANBertForMaskedLM(config).to(device)
    
    print("\nApplying torch.compile for ultimate performance...")
    compiled_model = torch.compile(model, mode="max-autotune")
    dataset = HybridStreamingMLMDataset(args.data_path, tokenizer, args.seq_len)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-6)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(train_loader) * args.epochs)

    print("\nStarting ultimate performance training...")
    for epoch in range(args.epochs):
        train_one_epoch(compiled_model, train_loader, optimizer, scheduler, device, epoch, args.epochs, use_amp)
        save_checkpoint(compiled_model, optimizer, scheduler, epoch, args.output_dir)
    
    print("\nTraining finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ultimate Performance CAN-BERT Pre-training with Shared Memory & JIT")
    
    # 필수 인자
    parser.add_argument("--data_path", type=str, required=True, help="Path to the candump log file.")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to save/load the vocab.json file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints.")
    
    # 훈련 하이퍼파라미터
    parser.add_argument("--seq_len", type=int, default=126, help="Sequence length.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size per GPU.")
    parser.add_argument("--epochs", type=int, default=20, help="Total training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Peak learning rate.")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps for learning rate scheduler.")
    
    # 모델 아키텍처
    parser.add_argument("--hidden_size", type=int, default=512, help="Model hidden size (d_model).")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--intermediate_size", type=int, default=1024, help="Size of the FFN intermediate layer.")
    parser.add_argument("--dataset_type", type=str, default="candump", help="Type of dataset format to parse (e.g., candump, csv).")

    # 시스템 설정
    parser.add_argument("--num_workers", type=int, default=min(16, os.cpu_count()), help="DataLoader worker processes.")
    
    args = parser.parse_args()
    
    # 스크립트가 직접 실행될 때만 main 함수 호출
    # (멀티프로세싱 시 자식 프로세스가 이 부분을 실행하는 것을 방지)
    if mp.get_start_method(allow_none=True) != 'fork':
        mp.set_start_method('fork', force=True)
        
    main(args)