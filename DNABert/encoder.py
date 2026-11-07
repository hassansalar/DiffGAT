"""
DNABert encoder with robust loading for DNABERT-2 and rich k-mer fallback.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from dnabert.embedding import kmer_counts

logger = logging.getLogger(__name__)


class DNABertEncoder:
    def __init__(
        self,
        model_name: str = "zhihan1996/DNABERT-2-117M",
        max_len: int = 36,
        batch_size: int = 128,
        pooling: str = "mean",
        device: str = "cuda",
        use_amp: bool = True,
        cache_dir: Optional[str | Path] = None,
        kmer_k: int = 5,
    ) -> None:
        self.model_name = model_name
        self.max_len = max_len
        self.batch_size = batch_size
        self.pooling = pooling
        self.device = device
        self.use_amp = use_amp and str(device).startswith("cuda")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.kmer_k = kmer_k

        self.tokenizer = None
        self.model = None
        self.embedding_dim = 4 ** kmer_k  # fallback
        self.use_fallback = True

    def load_model(self) -> None:
        logger.info("Loading DNABert: %s", self.model_name)
        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            cfg = AutoConfig.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            if getattr(cfg, "pad_token_id", None) is None:
                cfg.pad_token_id = 3
            if not hasattr(cfg, "is_decoder"):
                cfg.is_decoder = False

            model = None
            last_err = None
            # Several strategies to avoid meta-device / accelerate issues
            strategies = [
                dict(trust_remote_code=True, low_cpu_mem_usage=False,
                     torch_dtype=torch.float32),
                dict(trust_remote_code=True, low_cpu_mem_usage=True,
                     torch_dtype=torch.float32, device_map="cpu"),
                dict(trust_remote_code=True, low_cpu_mem_usage=False),
            ]
            for i, kwargs in enumerate(strategies):
                try:
                    model = AutoModel.from_pretrained(
                        self.model_name, config=cfg, **kwargs
                    )
                    break
                except Exception as e:
                    last_err = e
                    logger.warning("   AutoModel attempt %d failed: %s", i + 1, e)
                    model = None
                try:
                    from transformers import AutoModelForMaskedLM
                    model = AutoModelForMaskedLM.from_pretrained(
                        self.model_name, config=cfg, **kwargs
                    )
                    break
                except Exception as e:
                    last_err = e
                    logger.warning(
                        "   AutoModelForMaskedLM attempt %d failed: %s", i + 1, e
                    )
                    model = None

            if model is None:
                raise RuntimeError(f"All loaders failed: {last_err}")

            if hasattr(model, "bert"):
                model = model.bert
            elif hasattr(model, "model") and not hasattr(model, "embeddings"):
                model = model.model

            # Materialise any residual meta tensors then move to device
            try:
                if any(getattr(p, "is_meta", False) for p in model.parameters()):
                    model = model.to_empty(device="cpu")
            except Exception:
                pass
            self.model = model.to(self.device).eval()
            self.embedding_dim = (
                getattr(self.model.config, "hidden_size", None)
                or getattr(self.model.config, "d_model", 768)
            )
            self.use_fallback = False
            logger.info("DNABert loaded (dim=%d)", self.embedding_dim)
        except Exception as e:
            logger.warning(
                "DNABert load failed: %s → using %d-mer fallback encoder (%d-d)",
                e, self.kmer_k, 4 ** self.kmer_k,
            )
            self.model = self.tokenizer = None
            self.embedding_dim = 4 ** self.kmer_k
            self.use_fallback = True

    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        sequences = list(sequences)
        uniq = list(dict.fromkeys(sequences))
        pos = {s: i for i, s in enumerate(uniq)}
        inverse = np.fromiter(
            (pos[s] for s in sequences), dtype=np.int64, count=len(sequences)
        )
        if len(uniq) < len(sequences):
            logger.info(
                "   %s sequences, %s unique",
                f"{len(sequences):,}",
                f"{len(uniq):,}",
            )

        cache_file = self._cache_path(uniq)
        if cache_file is not None and cache_file.exists():
            logger.info("   Loaded cached embeddings: %s", cache_file.name)
            return np.load(cache_file)[inverse]

        if self.use_fallback or self.model is None:
            emb = kmer_counts(uniq, k=self.kmer_k)
            self.embedding_dim = emb.shape[1]
        else:
            try:
                emb = self._encode_model(uniq)
            except Exception as e:
                logger.warning(
                    "DNABert encoding failed (%s) → %d-mer fallback for ALL sequences",
                    e, self.kmer_k,
                )
                self.use_fallback = True
                emb = kmer_counts(uniq, k=self.kmer_k)
                self.embedding_dim = emb.shape[1]

        if cache_file is not None:
            np.save(cache_file, emb)
        return emb[inverse]

    def _cache_path(self, uniq: List[str]) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        kind = (
            f"kmer{self.kmer_k}"
            if (self.use_fallback or self.model is None)
            else self.model_name
        )
        h = hashlib.md5()
        h.update(f"{kind}|{self.pooling}|{self.max_len}|".encode())
        h.update("\n".join(uniq).encode())
        return self.cache_dir / f"emb_{h.hexdigest()[:16]}.npy"

    def _encode_model(self, uniq: List[str]) -> np.ndarray:
        order = np.argsort([len(s) for s in uniq], kind="stable")
        out = np.empty((len(uniq), self.embedding_dim), dtype=np.float32)
        bs, i = self.batch_size, 0
        pbar = tqdm(total=len(uniq), desc="DNABert")
        while i < len(uniq):
            sel = order[i : i + bs]
            batch = [uniq[j] for j in sel]
            try:
                out[sel] = self._encode_batch(batch)
            except torch.cuda.OutOfMemoryError:
                if bs == 1:
                    raise
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                logger.warning("   CUDA OOM → batch size %d", bs)
                continue
            except Exception:
                if not self.use_amp:
                    raise
                logger.warning("   fp16 autocast failed → retrying in fp32")
                self.use_amp = False
                continue
            i += len(sel)
            pbar.update(len(sel))
        pbar.close()
        return out

    @torch.inference_mode()
    def _encode_batch(self, batch: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        dev_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        with torch.autocast(
            device_type=dev_type, dtype=torch.float16, enabled=self.use_amp
        ):
            outputs = self.model(**inputs)

        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        elif getattr(outputs, "hidden_states", None) is not None:
            hidden = outputs.hidden_states[-1]
        else:
            hidden = outputs[0]
        hidden = hidden.float()

        mask = inputs.get("attention_mask")
        if mask is None:
            mask = torch.ones(hidden.shape[:2], device=hidden.device)
        mask = mask.unsqueeze(-1).to(hidden.dtype)

        if self.pooling == "max":
            emb = hidden.masked_fill(mask == 0, float("-inf")).max(dim=1).values
        elif self.pooling == "cls":
            emb = hidden[:, 0, :]
        else:
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return emb.cpu().numpy()
