from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config_region_mm import RegionAwareMMConfig
from .fusion_mapper import PrefixFusionMapper
from .param_encoder import NumericParamEncoder
from .projector import DefectProjector
from .prompt_builder_region import build_messages, build_training_prompt, build_inference_prompt
from .region_prior import RegionPriorEncoder
from .vision_backbone import YOLOFeatureBackbone, roi_pool_from_boxes


def prepare_model_for_kbit_training_no_fp32_cast(model):
    """
    4B/量化模型 LoRA 训练用的轻量版 prepare。

    目的：
    1. 避免 peft.prepare_model_for_kbit_training 把部分参数强制转 fp32 导致显存爆掉；
    2. 冻结 base LLM；
    3. 开启 gradient checkpointing；
    4. 关闭 use_cache；
    5. 开启输入 embedding 梯度，保证 LoRA / 多模态前缀可以正常反传。

    注意：
    - 这个函数不会把 LayerNorm / embedding 强制转 float32；
    - 适合 Qwen3-4B 4bit + 第一次书本知识 LoRA 继续训练场景。
    """

    # 冻结底座大模型参数
    for param in model.parameters():
        param.requires_grad = False

    # 训练时关闭 cache，减少显存占用
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # 开启梯度检查点，降低激活显存
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    # 让输入 embedding 支持梯度回传
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, inputs, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return model


class RegionAwareMultiModalModel(nn.Module):
    def __init__(self, cfg: RegionAwareMMConfig):
        super().__init__()
        self.cfg = cfg
        self.runtime_device = torch.device(cfg.device)

        self.vision = YOLOFeatureBackbone(
            weight_path=cfg.yolo_weight_path,
            feature_layer=cfg.yolo_feature_layer,
            device=cfg.device,
        )

        if cfg.freeze_yolo:
            if hasattr(self.vision, "set_trainable"):
                self.vision.set_trainable(False)
            else:
                for p in self.vision.parameters():
                    p.requires_grad = False

        self.tokenizer = self._build_tokenizer(cfg)
        self.llm = self._build_llm(cfg)

        if hasattr(self.llm, "gradient_checkpointing_enable"):
            try:
                self.llm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.llm.gradient_checkpointing_enable()

        if hasattr(self.llm, "enable_input_require_grads"):
            self.llm.enable_input_require_grads()

        if hasattr(self.llm, "config") and hasattr(self.llm.config, "use_cache"):
            self.llm.config.use_cache = False

        embed_layer = self.llm.get_input_embeddings()
        embed_dim = embed_layer.embedding_dim
        self.llm_embed_dtype = embed_layer.weight.dtype

        self.projector = DefectProjector(
            cfg.vision_feat_dim,
            cfg.projector_hidden_dim,
            embed_dim,
            cfg.projector_dropout,
        ).to(self.runtime_device)

        self.param_encoder = NumericParamEncoder(
            len(cfg.numeric_fields),
            embed_dim,
            cfg.numeric_token_count,
        ).to(self.runtime_device)

        self.region_prior = RegionPriorEncoder(
            region_order=cfg.region_order,
            embed_dim=embed_dim,
            hidden_dim=cfg.region_hidden_dim,
            view_vocab_size=cfg.view_vocab_size,
            use_external_summary_tokens=cfg.use_external_summary_tokens,
            use_coordinate_region_summary=cfg.use_coordinate_region_summary,
        ).to(self.runtime_device)

        self.prefix_fusion = None
        if cfg.use_prefix_fusion_mapper and cfg.fusion_token_count > 0:
            self.prefix_fusion = PrefixFusionMapper(
                embed_dim=embed_dim,
                token_count=cfg.fusion_token_count,
                num_heads=cfg.fusion_num_heads,
                dropout=cfg.projector_dropout,
            ).to(self.runtime_device)

    def _build_tokenizer(self, cfg: RegionAwareMMConfig):
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.llm_base_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "left"
        return tokenizer

    def _build_llm(self, cfg: RegionAwareMMConfig):
        if cfg.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quant_config = None

        llm = AutoModelForCausalLM.from_pretrained(
            cfg.llm_base_path,
            quantization_config=quant_config,
            trust_remote_code=True,
            local_files_only=True,
            device_map={"": 0} if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )

        # =============================================================
        # 推荐路线：base + 第一次微调 LoRA adapter（书本知识，未合并）
        # 然后继续训练这个 adapter 和多模态新增模块。
        #
        # 关键修改：
        # 不再使用 peft.prepare_model_for_kbit_training(llm)
        # 改用 prepare_model_for_kbit_training_no_fp32_cast(llm)
        # 避免不必要的 fp32 cast，并保持 base 冻结、LoRA 可训练。
        # =============================================================
        if getattr(cfg, "llm_stage1_adapter_path", "") and getattr(cfg, "continue_train_stage1_adapter", False):
            if cfg.load_in_4bit:
                llm = prepare_model_for_kbit_training_no_fp32_cast(llm)

            llm = PeftModel.from_pretrained(
                llm,
                cfg.llm_stage1_adapter_path,
                is_trainable=True,
            )
            print(f"[llm] loaded stage1 LoRA adapter as trainable: {cfg.llm_stage1_adapter_path}")

        elif cfg.llm_stage2_adapter_path and cfg.continue_train_stage2_adapter:
            if cfg.load_in_4bit:
                llm = prepare_model_for_kbit_training_no_fp32_cast(llm)

            llm = PeftModel.from_pretrained(
                llm,
                cfg.llm_stage2_adapter_path,
                is_trainable=True,
            )
            print(f"[llm] loaded stage2 LoRA adapter as trainable: {cfg.llm_stage2_adapter_path}")

        elif cfg.train_new_lora_adapter:
            if cfg.load_in_4bit:
                llm = prepare_model_for_kbit_training_no_fp32_cast(llm)

            lora_cfg = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(cfg.lora_target_modules),
            )
            llm = get_peft_model(llm, lora_cfg)
            print("[llm] initialized a new trainable LoRA adapter")

        elif cfg.freeze_llm_base:
            for p in llm.parameters():
                p.requires_grad = False
            print("[llm] base model frozen, no LoRA adapter is trainable")

        if hasattr(llm, "print_trainable_parameters"):
            llm.print_trainable_parameters()

        return llm

    def _select_view_indices(self, view_items: Sequence[Dict[str, Any]]) -> List[int]:
        if not view_items:
            return []

        scored = []
        for idx, item in enumerate(view_items):
            defect_cnt = float(item.get("total_defect_count", 0) or 0)
            area = float(item.get("total_area_mm2", 0) or 0)
            score = defect_cnt * 1000.0 + area
            scored.append((score, idx))

        scored.sort(key=lambda x: (-x[0], x[1]))
        keep = [idx for _, idx in scored[: self.cfg.max_views_per_sample]]
        keep.sort()
        return keep

    def _cap_prefix_tokens(self, prefix_tokens: torch.Tensor) -> torch.Tensor:
        if prefix_tokens.size(0) > self.cfg.max_prefix_tokens:
            prefix_tokens = prefix_tokens[: self.cfg.max_prefix_tokens]
        return prefix_tokens

    def _compose_prefix_tokens(self, token_groups: Sequence[torch.Tensor]) -> torch.Tensor:
        prefix_tokens = torch.cat(list(token_groups), dim=0)

        if self.prefix_fusion is not None:
            raw_budget = max(1, self.cfg.max_prefix_tokens - self.cfg.fusion_token_count)
            if prefix_tokens.size(0) > raw_budget:
                prefix_tokens = prefix_tokens[:raw_budget]

            fused_tokens = self.prefix_fusion(prefix_tokens)
            prefix_tokens = torch.cat([fused_tokens, prefix_tokens.float()], dim=0)

        prefix_tokens = self._cap_prefix_tokens(prefix_tokens)
        return prefix_tokens.to(dtype=self.llm_embed_dtype, device=self.runtime_device)

    def encode_sample_visual_tokens(
        self,
        images: Sequence[torch.Tensor],
        boxes_by_view: Sequence[Sequence[Sequence[float]]],
        view_items: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        selected = self._select_view_indices(view_items)
        if not selected:
            raise ValueError("当前样本没有可用视角。")

        tokens: List[torch.Tensor] = []
        for idx in selected:
            image = images[idx].unsqueeze(0).to(self.runtime_device)
            boxes = boxes_by_view[idx] if idx < len(boxes_by_view) else []

            if self.cfg.freeze_yolo:
                with torch.no_grad():
                    vision_out = self.vision(image)
            else:
                vision_out = self.vision(image)

            roi_tokens = roi_pool_from_boxes(
                vision_out.feature_map,
                boxes_xyxy=boxes,
                image_size=self.cfg.vision_image_size,
                output_size=self.cfg.roi_output_size,
                max_rois=self.cfg.max_rois_per_view,
            )

            tokens.append(roi_tokens.squeeze(0))

        vision_tokens = torch.cat(tokens, dim=0).to(self.runtime_device)
        vision_tokens = self.projector(vision_tokens)
        return vision_tokens.to(dtype=self.llm_embed_dtype, device=self.runtime_device)

    def encode_numeric_tokens(self, param_dict: Dict[str, Any]) -> torch.Tensor:
        values = [float(param_dict.get(k, 0.0) or 0.0) for k in self.cfg.numeric_fields]
        vec = torch.tensor(
            values,
            dtype=torch.float32,
            device=self.runtime_device,
        ).unsqueeze(0)

        out = self.param_encoder(vec).squeeze(0)
        return out.to(dtype=self.llm_embed_dtype, device=self.runtime_device)

    def encode_region_prior_tokens(
        self,
        summary: Dict[str, Any],
        defect_meta_by_view: Sequence[Sequence[Dict[str, Any]]],
        view_items: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        selected = self._select_view_indices(view_items)
        selected_meta = [
            defect_meta_by_view[idx] if idx < len(defect_meta_by_view) else []
            for idx in selected
        ]

        out = self.region_prior(
            summary=summary,
            defect_meta_by_view=selected_meta,
            image_size=self.cfg.vision_image_size,
            max_defect_tokens=self.cfg.max_defect_tokens,
            device=self.runtime_device,
        )
        return out.to(dtype=self.llm_embed_dtype, device=self.runtime_device)

    def _tokenize_pair(
        self,
        system_prompt: str,
        user_prompt: str,
        answer: str,
        text_budget: int,
    ) -> tuple[list[int], list[int]]:
        messages_user = build_messages(system_prompt, user_prompt, None)
        user_text = self.tokenizer.apply_chat_template(
            messages_user,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_messages = build_messages(system_prompt, user_prompt, answer)
        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        user_ids = self.tokenizer(user_text, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids
        answer_ids = full_ids[len(user_ids):]

        if len(answer_ids) > self.cfg.max_answer_tokens:
            answer_ids = answer_ids[: self.cfg.max_answer_tokens]

        remaining = max(64, min(self.cfg.max_user_tokens, text_budget - len(answer_ids)))
        if len(user_ids) > remaining:
            user_ids = user_ids[-remaining:]

        merged_ids = user_ids + answer_ids
        labels = [-100] * len(user_ids) + answer_ids
        return merged_ids, labels

    def _prepare_one(self, sample_idx: int, batch: Dict[str, Any]):
        summary = batch["summary"][sample_idx]
        params = batch["params"][sample_idx]
        question = batch["question"][sample_idx]
        answer = batch["answer"][sample_idx]
        workpiece_id = batch["workpiece_id"][sample_idx]
        instruction = batch["instruction"][sample_idx]
        defect_meta_by_view = batch["defect_meta_by_view"][sample_idx]
        view_items = batch["view_items"][sample_idx]
        info_text = batch["info_text"][sample_idx]

        visual_tokens = self.encode_sample_visual_tokens(
            batch["images"][sample_idx],
            batch["boxes_by_view"][sample_idx],
            view_items,
        )

        region_tokens = self.encode_region_prior_tokens(
            summary,
            defect_meta_by_view,
            view_items,
        )

        numeric_tokens = self.encode_numeric_tokens(params)

        prefix_tokens = self._compose_prefix_tokens(
            [visual_tokens, region_tokens, numeric_tokens]
        )

        user_prompt = build_training_prompt(
            workpiece_id=workpiece_id,
            summary=summary,
            params=params,
            question=question,
            view_items=view_items,
            info_text=info_text,
            include_visual_summary=self.cfg.use_visual_summary_text,
        )

        text_budget = max(192, self.cfg.max_seq_length - prefix_tokens.size(0))

        input_ids, text_labels = self._tokenize_pair(
            instruction,
            user_prompt,
            answer,
            text_budget=text_budget,
        )

        input_ids_t = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=self.runtime_device,
        )

        text_embeds = self.llm.get_input_embeddings()(input_ids_t)
        text_embeds = text_embeds.to(
            dtype=self.llm_embed_dtype,
            device=self.runtime_device,
        )

        prefix_labels = torch.full(
            (prefix_tokens.size(0),),
            -100,
            dtype=torch.long,
            device=self.runtime_device,
        )

        labels_t = torch.tensor(
            text_labels,
            dtype=torch.long,
            device=self.runtime_device,
        )

        inputs_embeds = torch.cat([prefix_tokens, text_embeds], dim=0)
        inputs_embeds = inputs_embeds.to(
            dtype=self.llm_embed_dtype,
            device=self.runtime_device,
        )

        attention_mask = torch.ones(
            inputs_embeds.size(0),
            dtype=torch.long,
            device=self.runtime_device,
        )

        labels = torch.cat([prefix_labels, labels_t], dim=0)

        return inputs_embeds, attention_mask, labels

    def _batch_to_inputs(self, batch: Dict[str, Any]):
        embed_list = []
        mask_list = []
        label_list = []

        for i in range(len(batch["workpiece_id"])):
            emb, mask, lab = self._prepare_one(i, batch)
            embed_list.append(emb)
            mask_list.append(mask)
            label_list.append(lab)

        max_len = max(x.size(0) for x in embed_list)
        hidden = embed_list[0].size(-1)
        bsz = len(embed_list)

        padded_embeds = torch.zeros(
            (bsz, max_len, hidden),
            dtype=self.llm_embed_dtype,
            device=self.runtime_device,
        )

        padded_mask = torch.zeros(
            (bsz, max_len),
            dtype=torch.long,
            device=self.runtime_device,
        )

        padded_labels = torch.full(
            (bsz, max_len),
            -100,
            dtype=torch.long,
            device=self.runtime_device,
        )

        for i, (emb, mask, lab) in enumerate(zip(embed_list, mask_list, label_list)):
            seq = emb.size(0)
            padded_embeds[i, :seq] = emb
            padded_mask[i, :seq] = mask
            padded_labels[i, :seq] = lab

        return padded_embeds, padded_mask, padded_labels

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        inputs_embeds, attention_mask, labels = self._batch_to_inputs(batch)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )

        return {"loss": outputs.loss}

    @torch.no_grad()
    def generate_one(
        self,
        workpiece_id: str,
        instruction: str,
        images: Sequence[torch.Tensor],
        boxes_by_view: Sequence[Sequence[Sequence[float]]],
        defect_meta_by_view: Sequence[Sequence[Dict[str, Any]]],
        summary: Dict[str, Any],
        params: Dict[str, Any],
        question: str,
        view_items: Sequence[Dict[str, Any]] | None = None,
        info_text: str = "",
        graph_evidence: Any = None,
        max_new_tokens: int = 256,
    ) -> str:
        view_items = list(view_items or [])

        visual_tokens = self.encode_sample_visual_tokens(
            images,
            boxes_by_view,
            view_items,
        )

        region_tokens = self.encode_region_prior_tokens(
            summary,
            defect_meta_by_view,
            view_items,
        )

        numeric_tokens = self.encode_numeric_tokens(params)

        prefix_tokens = self._compose_prefix_tokens(
            [visual_tokens, region_tokens, numeric_tokens]
        )

        user_prompt = build_inference_prompt(
            workpiece_id=workpiece_id,
            summary=summary,
            params=params,
            question=question,
            view_items=view_items,
            info_text=info_text,
            graph_evidence=graph_evidence,
            include_visual_summary=self.cfg.use_visual_summary_text,
        )

        messages = build_messages(instruction, user_prompt, None)
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompt_ids = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max(128, self.cfg.max_seq_length - prefix_tokens.size(0)),
        ).input_ids.to(self.runtime_device)

        text_embeds = self.llm.get_input_embeddings()(prompt_ids).squeeze(0)
        text_embeds = text_embeds.to(
            dtype=self.llm_embed_dtype,
            device=self.runtime_device,
        )

        inputs_embeds = torch.cat([prefix_tokens, text_embeds], dim=0).unsqueeze(0)

        attention_mask = torch.ones(
            inputs_embeds.size()[:2],
            dtype=torch.long,
            device=self.runtime_device,
        )

        output_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)