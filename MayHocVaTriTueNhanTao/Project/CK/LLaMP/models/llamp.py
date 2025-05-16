
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import LoraConfig, TaskType, LoraModel
from typing import Any, Optional, Tuple, Union

import os
import numpy as np

from flags import DATA_FOLDER

import math

from transformers import CLIPProcessor, CLIPModel
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

import copy
import os
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")


def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


class LLaMP(nn.Module):
    '''
    dset: dataset
    classnames: Dictionary chứa tên các lớp cho tập "base" và "new"
    args: Các tham số cấu hình từ
    model: Mô hình LLM đã được tải trước => meta-llama/Llama-2-7b-chat-hf
    tokenizer: Tokenizer tương ứng với LLM => Unused
    few_shot: Cờ boolean chỉ định đang ở chế độ few-shot hay zero-shot.
    indices: Chỉ số để chọn một tập con các lớp khi ở chế độ few-shot.
    '''
    def __init__(
        self, dset, classnames, args, model, tokenizer, few_shot=False, indices=None
    ):
        '''
            - Tải và tích hợp các mô hình tiền huấn luyện mạnh mẽ (CLIP, LLM).
            - Tạo ra các learnable parameters (prompts, LoRA weights, biases)
            - Tận dụng dữ liệu tiền xử lý (KV cache, text embeddings) để tăng hiệu quả.
            - Cung cấp các cơ chế kiểm soát (đóng băng thành phần, LoRA) để quản lý quá trình học.
        '''
        
        super(LLaMP, self).__init__()
        self.args = args
        self.dset = dset

        self.naive_decoding = args.naive_decoding
        self.debug = args.debug

        '''
        # vision_peft_config: Cấu hình LoRA cho các lớp self-attention (q_proj, v_proj) của
        # bộ mã hóa thị giác (vision model) của CLIP.
        # Mục đích: tinh chỉnh các ma trận trọng số tương ứng với phép chiếu Query (q_proj)
            và phép chiếu Value (v_proj) trong các khối self-attention của layer
            - 
        '''
        vision_peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "layers.{}.self_attn.q_proj".format(i)
                for i in range(args.v_lora_start, args.v_lora_end)
            ]
            + [
                "layers.{}.self_attn.v_proj".format(i)
                for i in range(args.v_lora_start, args.v_lora_end)
            ],
        )

        ### Cấu hình LoRA cho các lớp v_proj và q_proj của các decoder LLM được sử dụng 
        language_peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["v_proj", "q_proj"],
        )

        ### Tải mô hình CLIP tiền huấn luyệ
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch16",
            use_auth_token=hf_token,
        )

        # Đóng băng toàn bộ mô hình CLIP ban đầu
        self.clip_model.requires_grad_(False)

        # Tải CLIPProcessor tương ứng
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch16",
            use_auth_token=hf_token,
        )


        ### Chuẩn bị Input Texts cho CLIP:
        self.text_inputs = {}
        self.prompt_offset_indices = {}
        self.eos_offset = {}

        self.few_shot = few_shot

        self.num_prior_tokens = args.num_prior_tokens
        self.num_llm_prompts = args.num_llm_prompts
        self.num_text_template = args.num_text_template

        self.num_text_ctx = args.num_text_ctx
        self.llm_prompt_depth = args.llm_prompt_depth

        self.decoder_skip_connection = args.decoder_skip_connection
        self.concat_fixed_prompts = args.concat_fixed_prompts

        if self.concat_fixed_prompts:
            self.num_special_tokens = 4 + self.num_llm_prompts
        else:
            self.num_special_tokens = self.num_llm_prompts

        self.prompt_type = args.prompt_type

        target_list = ["base", "new"] if not self.few_shot else ["all"]

        '''
        - Lặp qua các tập mục tiêu (target_list: "base", "new" hoặc "all").
        - Mục tiêu cuối cùng là tạo ra các input_ids và attention_mask có "chỗ trống"
            (được đánh dấu bằng các token placeholder như BOS hoặc PAD ban đầu) để sau 
            này, trong quá trình forward của mô hình, các embedding của prompt học 
            được (cả từ LLM và các text prompt cho CLIP) có thể được chèn vào đúng vị 
            trí trong chuỗi đầu vào của bộ mã hóa văn bản CLIP.

        '''
        for target in target_list:
            # Với mỗi tập, tạo các prompt cơ bản dạng "a photo of a {classname}" cho từng lớp.
            # self.processor ở đây là CLIPProcessor, nó sẽ sử dụng tokenizer của CLIP
            # để chuyển đổi danh sách các chuỗi prompt thành input_ids và attention_mask.
            # self.text_inputs[target] giờ là một dictionary chứa 'input_ids' và 'attention_mask'.
            self.text_inputs[target] = self.processor(
                [
                    "a photo of a {}".format(c.replace("_", " "))
                    for c in classnames[target]
                ],
                return_tensors="pt",
                padding=True,
            )

            # Nếu loại prompt là "prefix" (các prompt học được sẽ được chèn vào đầu)
            if self.prompt_type == "prefix":
                self.text_inputs[target]["input_ids"] = torch.cat(
                    (
                        # Tạo một tensor gồm các token của CLIP tokenizer.
                        # Kích thước: (số_lớp, self.num_special_tokens)
                        torch.ones(
                            (len(classnames[target]), self.num_special_tokens),
                            dtype=torch.long,
                        )
                        * self.processor.tokenizer.bos_token_id,
                        self.text_inputs[target].input_ids,
                    ),
                    dim=1,
                )
                # => Kết quả: input_ids mới có dạng [BOS, BOS, ..., "a photo of a {classname}"]

                # Chèn placeholder vào 'attention_mask'
                self.text_inputs[target]["attention_mask"] = torch.cat(
                    (
                        # Tạo một tensor gồm các số 1.
                        # Kích thước: (số_lớp, self.num_special_tokens)
                        torch.ones(
                            (len(classnames[target]), self.num_special_tokens),
                            dtype=torch.long,
                        ),
                        self.text_inputs[target].attention_mask,
                    ),
                    dim=1,
                )
                # => Kết quả: attention_mask mới cũng có các số 1 ở đầu tương ứng với các token BOS.
            elif self.prompt_type == "suffix":
                # Suffix
                eos_loc = self.text_inputs[target]["input_ids"].argmax(dim=-1)
                idx = eos_loc != (self.text_inputs[target]["input_ids"].shape[1] - 1)

                self.text_inputs[target]["attention_mask"][:, -1] = 1
                self.text_inputs[target]["input_ids"] = torch.cat(
                    (
                        self.text_inputs[target].input_ids,
                        torch.ones(
                            (len(classnames[target]), self.num_special_tokens),
                            dtype=torch.long,
                        )
                        * self.processor.tokenizer.pad_token_id,
                    ),
                    dim=1,
                )
                self.text_inputs[target]["attention_mask"] = torch.cat(
                    (
                        self.text_inputs[target].attention_mask,
                        torch.ones(
                            (len(classnames[target]), self.num_special_tokens),
                            dtype=torch.long,
                        ),
                    ),
                    dim=1,
                )

                eos_loc = self.text_inputs[target]["input_ids"].argmax(dim=-1)
                self.text_inputs[target]["attention_mask"][
                    torch.arange(len(classnames[target]))[idx], eos_loc[idx]
                ] = 0

                self.eos_offset[target] = (
                    torch.arange(len(classnames[target])),
                    eos_loc,
                )

        self.eos_token_id = self.clip_model.text_model.eos_token_id

        # Một cờ cấu hình, có thể ảnh hưởng đến cách vision model được xử lý
        if self.naive_decoding:
            if args.freeze_vit:
                # Nếu cờ args.freeze_vit là True (đóng băng toàn bộ Vision Transformer của CLIP)
                self.lora_model = nn.ModuleDict(
                    {"default": self.clip_model.vision_model} # Sử dụng vision_model gốc của CLIP
                )
                self.lora_model.requires_grad_(False)
            else:
                # Nếu không đóng băng toàn bộ ViT, thì áp dụng LoRA cho nó
                self.lora_model = nn.ModuleDict(
                    {
                        "default": LoraModel(
                            self.clip_model.vision_model,
                            {"default": vision_peft_config}, # Sử dụng cấu hình LoRA cho vision đã định nghĩa ở trên
                            "default",
                        )
                    }
                )

        # Lấy kích thước ẩn của mô hình văn bản CLIP
        self.text_hidden_size = self.clip_model.text_model.config.hidden_size

        print("Loading CLIP text embeddings from {}".format(args.clip_text_embed_file))
        
        ### Tải các text embeddings của CLIP đã được tính toán trước
        text_embeddings = torch.load(
            os.path.join(self.dset.data_dir, args.clip_text_embed_file)
        )

        # Chuyển các tensor embeddings thành nn.Parameter với requires_grad=False.
        if type(text_embeddings["base"]) == dict:
            self.base_embeddings = nn.Parameter(
                text_embeddings["base"]["avg"], requires_grad=False
            )
            self.new_embeddings = nn.Parameter(
                text_embeddings["new"]["avg"], requires_grad=False
            )
        else:
            self.base_embeddings = nn.Parameter(
                text_embeddings["base"], requires_grad=False
            )
            self.new_embeddings = nn.Parameter(
                text_embeddings["new"], requires_grad=False
            )

        if self.few_shot:
            # Tạo self.text_embeddings dựa trên chế độ few_shot và indices
            if indices is not None:
                # Nếu có cung cấp 'indices' (chỉ số của các lớp được chọn cho few-shot)
                self.text_embeddings = nn.ParameterDict(
                    {
                        "all": torch.cat(
                            (self.base_embeddings, self.new_embeddings), dim=0
                        )[indices]
                    }
                )
            else:
                # Nếu không có 'indices', sử dụng tất cả các lớp đã ghép
                self.text_embeddings = nn.ParameterDict(
                    {
                        "all": torch.cat(
                            (self.base_embeddings, self.new_embeddings), dim=0
                        )
                    }
                )
        else:
            # Nếu ở chế độ zero-shot => Lưu trữ riêng biệt cho base và new
            self.text_embeddings = nn.ParameterDict(
                {
                    "base": self.base_embeddings,
                    "new": self.new_embeddings,
                }
            )

        #  Khởi tạo các thuộc tính liên quan đến loss và bias
        self.distillation_type = args.distillation_type
        self.base_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        self.token_bias = args.token_bias

        # => True
        self.visual_prompting = args.visual_prompting

        # Xác định độ sâu tối đa của prompt
        self.prompt_depth = max(self.llm_prompt_depth, args.text_prompt_depth)

        self.visual_prompt_depth = args.visual_prompt_depth
        self.text_prompt_depth = self.prompt_depth

        ### Khởi tạo Visual Prompts nếu được kích hoạt
        if self.visual_prompting:
            self.visual_prompts = nn.Parameter(
                torch.empty(
                    (
                        self.visual_prompt_depth,
                        args.num_vis_ctx,
                        self.clip_model.vision_model.config.hidden_size,
                    )
                ).normal_(0, 1)
            )

            # Khởi tạo trọng số Kaiming Uniform cho visual prompts
            nn.init.kaiming_uniform_(self.visual_prompts, a=math.sqrt(5))

        ### Khởi tạo Text Prompts => "a photo of a"
        self.text_prompts = nn.ParameterList()

        # Chuỗi khởi tạo cho prompt đầu tiên
        ctx_init = "a photo of a"
        n_ctx = 4 # Số lượng context tokens cho prompt này ("a", "photo", "of", "a")
        
        # Tokenize chuỗi khởi tạo bằng CLIPProcessor
        prompt = self.processor([ctx_init], return_tensors="pt")
        
        # Không tính gradient cho bước này
        with torch.no_grad():
            # Lấy embedding của các token từ lớp embedding của mô hình văn bản CLIP
            embedding = self.clip_model.text_model.embeddings(
                input_ids=prompt.input_ids
            )

        # Lấy embedding của n_ctx token đầu tiên
        init_prompt = nn.Parameter(embedding[0, 1 : 1 + n_ctx, :], requires_grad=True)

        # Thêm prompt này vào danh sách
        self.text_prompts.extend(nn.ParameterList([init_prompt]))

        # Khởi tạo Text Prompts (phần 2: in-layer prompts)
        # Các prompt này sẽ được chèn vào các lớp sâu hơn của CLIP text encoder.
        self.in_layer_prompts = nn.ParameterList(
            [
                nn.Parameter(
                    # Kích thước: (num_text_ctx, text_hidden_size)
                    torch.empty(self.num_text_ctx, 512).normal_(0, 1),
                    requires_grad=True,
                )
                for _ in range(self.text_prompt_depth - 1) # Số lượng in-layer prompts = độ sâu - 1
            ]
        )

        for i in range(len(self.in_layer_prompts)):
            # # Khởi tạo Kaiming
            nn.init.kaiming_uniform_(self.in_layer_prompts[i], a=math.sqrt(5))

        # Ghép các in-layer prompts vào self.text_prompts
        self.text_prompts.extend(self.in_layer_prompts)

        self.num_decoder_layers = args.num_decoder_layers

        print("Loading past key values from {}".format(args.past_key_value_file))
        ### Tải Past Key Values (KV Cache) từ file
        content_dict = torch.load(
            os.path.join(DATA_FOLDER, dset.data_dir, args.past_key_value_file)
        )
        
        # Tạo 'all' key-values nếu ở chế độ few_shot và 'all' chưa có
        if few_shot and "all" not in content_dict:
            print("Auto-creating 'all' from 'base' and 'new'")
            content_dict["all"] = content_dict["base"] + content_dict["new"]

        '''
        - Tạo self.past_key_values và self.attention_mask (và self.next_token_bias nếu có argument)
        - Ba biến này sẽ được sử dụng trong hàm `decode_class` để cung cấp context cho các lớp decoder LLM.

        '''
        if self.few_shot:
            # Xử lý cho chế độ few_shot (sử dụng content_dict["all"])
            if indices is not None:
                # Nếu có 'indices', chọn KV cache và attention mask tương ứng với các lớp được chọn
                self.all_class_key_values = nn.ParameterList(
                    [
                        nn.Parameter(
                            x["past_key_values"][
                                -self.num_decoder_layers :, :, indices
                            ],
                            requires_grad=False,
                        )
                        for x in content_dict["all"]
                    ]
                )
                self.all_class_attn_mask = [
                    x["attn_mask"][indices] for x in content_dict["all"]
                ]
            else:
                # Nếu không có 'indices', lấy toàn bộ KV cache và attention mask cho 'all'
                self.all_class_key_values = nn.ParameterList(
                    [
                        nn.Parameter(
                            x["past_key_values"][-self.num_decoder_layers :],
                            requires_grad=False,
                        )
                        for x in content_dict["all"]
                    ]
                )
                self.all_class_attn_mask = [x["attn_mask"] for x in content_dict["all"]]

            # # Gán vào dictionary với key "all"
            self.past_key_values = nn.ParameterDict(
                {
                    "all": self.all_class_key_values,
                }
            )

            # Lưu trữ attention mask tương ứng
            self.attention_mask = {
                "all": self.all_class_attn_mask,
            }
        else:
            # Xử lý cho chế độ zero-shot (sử dụng content_dict["base"] và content_dict["new"])
            self.base_class_key_values = nn.ParameterList(
                [
                    nn.Parameter(
                        x["past_key_values"][-self.num_decoder_layers :],
                        requires_grad=False,
                    )
                    for x in content_dict["base"]
                ]
            )
            self.base_class_attn_mask = [x["attn_mask"] for x in content_dict["base"]]

            self.new_class_key_values = nn.ParameterList(
                [
                    nn.Parameter(
                        x["past_key_values"][-self.num_decoder_layers :],
                        requires_grad=False,
                    )
                    for x in content_dict["new"]
                ]
            )
            self.new_class_attn_mask = [x["attn_mask"] for x in content_dict["new"]]

            self.past_key_values = nn.ParameterDict(
                {
                    "base": self.base_class_key_values,
                    "new": self.new_class_key_values,
                }
            )

            self.attention_mask = {
                "base": self.base_class_attn_mask,
                "new": self.new_class_attn_mask,
            }

            # Xử lý và lưu trữ Token Bias
            if self.token_bias:
                self.base_token_bias = nn.ParameterList(
                    [
                        nn.Parameter(
                            x["next_token_embeds"][
                                -self.num_decoder_layers, :, : self.num_prior_tokens, :
                            ],
                            requires_grad=False,
                        )
                        for x in content_dict["base"]
                    ]
                    if self.token_bias
                    else [torch.zeros(1)]
                )

                self.new_token_bias = nn.ParameterList(
                    [
                        nn.Parameter(
                            x["next_token_embeds"][
                                -self.num_decoder_layers, :, : self.num_prior_tokens, :
                            ],
                            requires_grad=False,
                        )
                        for x in content_dict["new"]
                    ]
                    if self.token_bias
                    else [torch.zeros(1)]
                )

                self.base_token_bias_attn_mask = [
                    x["next_token_attn_mask"] for x in content_dict["base"]
                ]
                self.new_token_bias_attn_mask = [
                    x["next_token_attn_mask"] for x in content_dict["new"]
                ]

                self.next_token_bias = nn.ParameterDict(
                    {
                        "base": self.base_token_bias,
                        "new": self.new_token_bias,
                    }
                )

                self.next_token_attn_mask = {
                    "base": self.base_token_bias_attn_mask,
                    "new": self.new_token_bias_attn_mask,
                }

        ### Khởi tạo Class Tokens (các learnable prompt cho LLM)
        self.class_token = nn.ParameterList(
            [
                nn.Parameter(
                    # Kích thước: (số_llm_prompts, kích_thước_ẩn_LLM)
                    torch.empty(
                        (self.num_llm_prompts, model.config.hidden_size)
                    ).normal_(0, 1)
                )
                for _ in range(1) 
                # Chỉ tạo 1 bộ class_token, bộ này sẽ được dùng cho tất cả các lớp
                # bằng cách expand trong hàm generate_text_features_from_prompt
            ]
        )
        
        for i in range(len(self.class_token)):
            nn.init.kaiming_uniform_(self.class_token[i], a=math.sqrt(5))

        ### Khởi tạo các Lớp Decoder của LLM (`self.class_decoder`) và Norm Layer ---
        self.class_proj = nn.Identity()
        self.class_norm = copy.deepcopy(model.model.norm)
        if args.lora_decoding:
            self.class_decoder = nn.ModuleList(
                [
                    LoraModel(
                        copy.deepcopy(model.model.layers[i]),
                        {"default": language_peft_config},
                        "default",
                    )
                    for i in range(-self.num_decoder_layers, 0)
                ]
            )
            self.class_norm.requires_grad_(False)
        else:
            self.class_decoder = nn.ModuleList(
                [
                    copy.deepcopy(model.model.layers[i])
                    for i in range(-self.num_decoder_layers, 0)
                ]
            )

            # Trong trường hợp này, các lớp decoder và norm layer có thể được huấn luyện
            self.class_decoder.requires_grad_(True)
            self.class_norm.requires_grad_(True)

        ### Khởi tạo các Lớp Chiếu Văn bản (`self.text_proj`)
        self.text_proj = nn.ModuleList(
            [
                # Lớp Linear để chiếu từ kích thước ẩn của LLM sang kích thước ẩn của CLIP text model.
                nn.Linear(model.config.hidden_size, self.text_hidden_size, bias=False)
                for _ in range(self.llm_prompt_depth)
            ]
        )

        ### Khởi tạo LLM Prompt Bias (`self.llm_prompt_bias`)
        self.llm_prompt_bias = nn.ParameterList(
            [
                # Kích thước: (self.num_special_tokens, text_hidden_size của CLIP)
                nn.Parameter(torch.empty(self.num_special_tokens, 512).normal_(0, 1))
                for _ in range(self.llm_prompt_depth)
            ]
        )

        for i in range(len(self.llm_prompt_bias)):
            nn.init.kaiming_uniform_(self.llm_prompt_bias[i], a=math.sqrt(5))

        ### Khởi tạo Trọng số cho Class Embeddings (`self.class_embed_weight`)
        self.class_embed_weight = nn.Parameter(torch.zeros(1), requires_grad=False)

        ### Kiểm soát Đóng băng Thành phần
        if args.learn_class_embed_weight:
            self.class_embed_weight.requires_grad_(True)

        if args.prompt_learning:
            self.class_decoder.requires_grad_(False)
            self.class_norm.requires_grad_(False)

        if args.freeze_decoder_kv_proj:
            for decoder in self.class_decoder:
                decoder.self_attn.k_proj.requires_grad_(False)
                decoder.self_attn.v_proj.requires_grad_(False)

        if args.freeze_decoder_q_proj:
            for decoder in self.class_decoder:
                decoder.self_attn.q_proj.requires_grad_(False)

        if args.freeze_decoder_o_proj:
            for decoder in self.class_decoder:
                decoder.self_attn.o_proj.requires_grad_(False)

        if args.freeze_decoder_attn:
            for decoder in self.class_decoder:
                decoder.self_attn.requires_grad_(False)

        if args.freeze_decoder_ffn:
            for decoder in self.class_decoder:
                decoder.mlp.requires_grad_(False)

        self.class_fn = self.decode_class


        ### Khởi tạo Logit Scale:
        self.logit_scale = nn.Parameter(
            torch.tensor([np.log(1 / 0.01)]), requires_grad=True
        )

        self.dropout = nn.Dropout(args.prompt_dropout)
        self.image_dropout = nn.Dropout(args.img_dropout)
        self.lambda_dist = args.lambda_dist

    '''
    Tạo ra các "adaptive prompts" bằng cách sử dụng kiến thức từ LLM.
    Các adaptive prompts này sau đó sẽ được đưa vào bộ mã hóa văn bản 
    của CLIP để tạo ra các đặc trưng văn bản cuối cùng cho các lớp đối tượng.
    '''
    def decode_class(self, subset="base", bias=None):
        ### Lấy Past Key Values (KV Cache) và Attention Mask tương ứng với subset
        
        # self.past_key_values là một nn.ParameterDict chứa KV cache cho các subset ("base", "new", hoặc "all").
        # pkv ở đây là một danh sách (List) các nn.Parameter, mỗi Parameter chứa KV cache cho một template prompt
        # đã được chạy qua các lớp đầu của LLM.
        pkv = self.past_key_values[subset]
        
        # self.attention_mask cũng là một dictionary chứa attention mask tương ứng với KV cache.
        # chỉ định những token nào trong KV cache cần được chú ý.
        attention_mask = self.attention_mask[subset]

        if self.training:
            '''
            Chọn ngẫu nhiên một template
                - self.num_text_template là tổng số template prompt mà KV cache đã được chuẩn bị.
                - Trong quá trình huấn luyện, mô hình sẽ ngẫu nhiên chọn một trong số các template này
                    để tạo adaptive prompts. Điều này giúp tăng tính đa dạng và khái quát hóa
            '''
            template_idx = torch.randint(self.num_text_template, (1,)).item()

            # Lấy Token Bias nếu được kích hoạt
            if self.token_bias:
                selected_embeddings = self.next_token_bias[subset][template_idx]
                selected_attn_mask = self.next_token_attn_mask[subset][template_idx]
            else:
                selected_embeddings = None
                selected_attn_mask = None

            # Tạo adaptive prompts từ prompt đã chọn
            encoded_prompt = self.generate_text_features_from_prompt(
                pkv[template_idx],              # KV cache của template đã chọn
                attention_mask[template_idx],   # Attention mask của template đã chọn
                self.class_token[0],            # Learnable class tokens
                selected_embeddings,            # Token bias (nếu có)
                selected_attn_mask,             # Attention mask cho token bias (nếu có)
                subset=subset,                  # Tên của subset hiện tại ("base", "new", "all")
            )
        else:
            # Khởi tạo một danh sách để lưu các adaptive prompts từ tất cả các template
            encoded_prompts = []

            # Lặp qua tất cả các template prompt
            for template_idx in range(self.num_text_template):

                if self.token_bias:
                    selected_embeddings = self.next_token_bias[subset][template_idx]
                    selected_attn_mask = self.next_token_attn_mask[subset][template_idx]
                else:
                    selected_embeddings = None
                    selected_attn_mask = None

                # Gọi hàm self.generate_text_features_from_prompt để tạo ra các embedding từ 
                # LLM dựa trên KV cache, class tokens, và token bias của template
                encoded_prompt = self.generate_text_features_from_prompt(
                    pkv[template_idx],
                    attention_mask[template_idx],
                    self.class_token[0],
                    selected_embeddings,
                    selected_attn_mask,
                    subset=subset,
                )
                encoded_prompts.append(encoded_prompt)

            encoded_prompt = torch.stack(encoded_prompts, dim=0)
        
        ### Trả về một tuple chứa:
        # - encoded_prompt: Các adaptive prompts đã được tạo ra và chiếu sang không gian của CLIP.
        # - self.text_embeddings[subset]: Các text embeddings gốc của CLIP cho subset đó (dùng để so sánh hoặc tính loss).

        outputs = ((encoded_prompt, self.text_embeddings[subset]),)
        return outputs

    ### Mục đích: Đây là hàm cốt lõi để LLM tạo ra các đặc trưng dựa trên KV Cache và các learnable class tokens.
    def generate_text_features_from_prompt(
        self,
        pkv,
        attention_mask,
        class_token,
        selected_embeddings=None,
        selected_attn_mask=None,
        subset="base",
    ):

        all_embeds = []

        ### Lấy class_token (learnable prompts cho LLM), mở rộng nó để có số lượng bằng số lớp trong subset.
        num_classes = self.text_embeddings[subset].shape[0]
        tokens = self.class_proj(class_token)
        tokens = tokens.unsqueeze(0).expand(num_classes, -1, -1)
        device = tokens.device

        
        if selected_embeddings is not None:
            # Nếu có selected_embeddings (token bias), ghép nó vào trước 
            # tokens và cập nhật attention_mask.
            attention_mask = torch.cat(
                (
                    attention_mask.to(device),
                    selected_attn_mask.to(device),
                    torch.ones((attention_mask.shape[0], tokens.shape[1])).to(device),
                ),
                dim=1,
            )

            tokens = torch.cat(
                [
                    selected_embeddings,
                    tokens,
                ],
                dim=1,
            )
        else:
            attention_mask = torch.cat(
                (
                    attention_mask.to(device),
                    torch.ones((attention_mask.shape[0], tokens.shape[1])).to(device),
                ),
                dim=1,
            )

        ### Chuẩn bị position_ids và attention_mask 4D cho các lớp decoder của LLM.
        position_ids = torch.clamp(
            torch.cumsum(attention_mask, dim=-1).long() - 1, min=0
        )[:, -tokens.shape[1] :]

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (num_classes, tokens.shape[1]), tokens, pkv.shape[-2]
        )

        hidden_states = tokens

        past_key_values_length = pkv[0][0].shape[2]

        ### Cho tokens (bao gồm class tokens và token bias nếu có) đi qua các 
        # lớp self.class_decoder (các lớp cuối của LLM). Quan trọng: Các lớp decoder
        # này sử dụng past_key_value=pkv[idx] từ KV Cache đã được tải trước.
        for idx, decoder_layer in enumerate(self.class_decoder):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=pkv[idx],
                use_cache=False,
                output_attentions=True,
            )
            hidden_states = layer_outputs[0]

        ### Áp dụng lớp chuẩn hóa self.class_norm.
        hidden_states = self.class_norm(hidden_states)

        ### Lấy các hidden states tương ứng với các vị trí của class_token (hoặc 
        # self.num_special_tokens nếu có token bias). Đây là class_embed.
        class_embed = hidden_states[:, -self.num_special_tokens :, :]

        # Áp dụng dropout và nhân với self.class_embed_weight.exp().
        class_embed = self.dropout(class_embed) * self.class_embed_weight.exp()

        # Lặp qua self.llm_prompt_depth: Chiếu class_embed qua self.text_proj[i] và cộng
        # với self.llm_prompt_bias[i]. Kết quả là các adaptive prompts ở các độ sâu khác nhau.
        for i in range(self.llm_prompt_depth):
            all_embeds.append(self.text_proj[i](class_embed) + self.llm_prompt_bias[i])

        # Gọi self.encode_LLM_prompt để đưa các adaptive prompts này vào bộ mã hóa văn bản của CLIP.
        encoded_prompt = self.encode_LLM_prompt(
            torch.stack(all_embeds, dim=0), subset=subset
        )
        return encoded_prompt

    ### Mục đích: Đưa các adaptive prompts (đã được chiếu từ LLM) và các text prompts cố 
    # định (ví dụ: "a photo of a") vào bộ mã hóa văn bản của CLIP để tạo ra đặc trưng 
    # văn bản cuối cùng.
    def encode_LLM_prompt(self, prompts, subset):
        device = prompts.device
        
        # Lấy input_ids và attention_mask của CLIP cho subset.
        input_ids = self.text_inputs[subset].input_ids.to(device)
        attention_mask = self.text_inputs[subset].attention_mask.to(device)
        
        position_ids = None

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        # Lấy embeddings ban đầu từ self.clip_model.text_model.embeddings.
        # Chèn adaptive prompts (prompts[0]) và text prompt đầu tiên (self.text_prompts[0]) vào 
        # vị trí thích hợp trong hidden_states (tùy theo self.prompt_type là "prefix" hay "suffix")
        if self.prompt_type == "prefix":
            hidden_states = self.clip_model.text_model.embeddings(
                input_ids=input_ids[:, self.num_special_tokens :],
                position_ids=position_ids,
            )
            hidden_states = torch.cat(
                [
                    hidden_states[:, :1, :],
                    torch.cat(
                        [
                            prompts[0],
                            self.text_prompts[0]
                            .unsqueeze(0)
                            .expand(hidden_states.shape[0], -1, -1),
                        ],
                        dim=1,
                    ),
                    hidden_states[:, 1 + self.num_text_ctx :, :],
                ],
                dim=1,
            )
        elif self.prompt_type == "suffix":
            hidden_states = self.clip_model.text_model.embeddings(
                input_ids=input_ids, position_ids=position_ids
            )
            hidden_states = torch.cat(
                [
                    hidden_states[:, :1, :],
                    self.text_prompts[0]
                    .unsqueeze(0)
                    .expand(hidden_states.shape[0], -1, -1),
                    hidden_states[
                        :, 1 + self.num_text_ctx : -self.num_special_tokens - 1, :
                    ],
                    prompts[0],
                    hidden_states[self.eos_offset[subset]].unsqueeze(1),
                ],
                dim=1,
            )

        ### Chuẩn bị causal_attention_mask và attention_mask 4D cho các lớp encoder của CLIP text model.
        causal_attention_mask = _make_causal_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        # - Lặp qua các lớp của self.clip_model.text_model.encoder.layers:
        #    + Nếu lớp hiện tại (idx) nằm trong khoảng text_prompt_depth: 
        #           + Chèn các adaptive prompts (prompts[idx] nếu idx < llm_prompt_depth) và text 
        #             prompts (self.text_prompts[idx]) tương ứng vào hidden_states.
        #    + Cho hidden_states đi qua lớp encoder hiện tại.
        for idx, encoder_layer in enumerate(self.clip_model.text_model.encoder.layers):
            if idx > 0 and idx < self.text_prompt_depth:
                if self.prompt_type == "prefix":
                    if idx < self.llm_prompt_depth:
                        next_prompts = torch.cat(
                            [
                                prompts[idx],
                                self.text_prompts[idx]
                                .unsqueeze(0)
                                .expand(hidden_states.shape[0], -1, -1),
                            ],
                            dim=1,
                        )
                    else:
                        next_prompts = torch.cat(
                            [
                                hidden_states[:, 1 : 1 + self.num_special_tokens, :],
                                self.text_prompts[idx]
                                .unsqueeze(0)
                                .expand(hidden_states.shape[0], -1, -1),
                            ],
                            dim=1,
                        )

                    hidden_states = torch.cat(
                        [
                            hidden_states[:, :1, :],
                            next_prompts,
                            hidden_states[
                                :, 1 + self.num_text_ctx + self.num_special_tokens :, :
                            ],
                        ],
                        dim=1,
                    )

                elif self.prompt_type == "suffix":
                    if idx < self.llm_prompt_depth:
                        hidden_states = torch.cat(
                            [
                                hidden_states[:, :1, :],
                                self.text_prompts[idx]
                                .unsqueeze(0)
                                .expand(hidden_states.shape[0], -1, -1),
                                hidden_states[
                                    :,
                                    1
                                    + self.num_text_ctx : -self.num_special_tokens
                                    - 1,
                                    :,
                                ],
                                prompts[idx],
                                hidden_states[:, -1:, :],
                            ],
                            dim=1,
                        )
                    else:
                        hidden_states = torch.cat(
                            [
                                hidden_states[:, :1, :],
                                self.text_prompts[idx]
                                .unsqueeze(0)
                                .expand(hidden_states.shape[0], -1, -1),
                                hidden_states[:, 1 + self.num_text_ctx :, :],
                            ],
                            dim=1,
                        )

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
            )

            hidden_states = layer_outputs[0]

        # Áp dụng lớp chuẩn hóa cuối cùng (self.clip_model.text_model.final_layer_norm).
        last_hidden_state = hidden_states
        last_hidden_state = self.clip_model.text_model.final_layer_norm(
            last_hidden_state
        )

        if self.prompt_type == "prefix":
            pooled_output = last_hidden_state[
                torch.arange(
                    last_hidden_state.shape[0], device=last_hidden_state.device
                ),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(
                    dim=-1
                ),
            ]
        else:
            pooled_output = last_hidden_state[:, -1, :]

        # Chiếu pooled output qua self.clip_model.text_projection để có được đặc trưng văn bản cuối cùng.
        text_features = self.clip_model.text_projection(pooled_output)

        return text_features

    def forward(self, x, subset=None):
        if self.training:
            loss, pred = self.run(x)
            # Nếu đang huấn luyện, trả về loss và dự đoán.
            return loss, pred
        else:
            scores = self.run(x, subset)
            # Trả về điểm số (logits hoặc probabilities).
            return None, scores

    '''
    Tính toán trước tất cả các class embeddings (cả từ LLM và CLIP gốc) cho một subset và 
    lưu vào self.all_class_embed. Được sử dụng trong quá trình đánh giá để không phải 
    tính toán lại nhiều lần.
    '''
    def compute_all_class_embeddings(self, subset):
        outputs = self.class_fn(subset=subset)
        class_embed = outputs[0]

        self.all_class_embed = class_embed

    '''
    Trích xuất đặc trưng hình ảnh từ bộ mã hóa hình ảnh của CLIP, có tích 
    hợp visual prompts và LoRA.

    '''
    def extract_image_features(self, img, target="default", dropout=False):
        if self.visual_prompting:
            image_features = self.extract_prompt_image_features(
                img, model=self.lora_model[target]
            )
        else:
            image_features = self.lora_model[target](img)[1]
            if dropout:
                image_features = self.image_dropout(image_features)
            image_features = self.clip_model.visual_projection(image_features)
        return image_features

    def extract_prompt_image_features(self, img, model, dropout=False):
        hidden_states = model.embeddings(img)
        hidden_states = torch.cat(
            [
                hidden_states,
                self.visual_prompts[0]
                .unsqueeze(0)
                .expand(hidden_states.shape[0], -1, -1),
            ],
            dim=1,
        )
        hidden_states = model.pre_layrnorm(hidden_states)

        len_vpt = self.visual_prompts.shape[1]

        for idx, encoder_layer in enumerate(model.encoder.layers):
            if idx > 0 and idx < self.visual_prompt_depth:
                hidden_states = torch.cat(
                    [
                        hidden_states[:, :-len_vpt],
                        self.visual_prompts[idx]
                        .unsqueeze(0)
                        .expand(hidden_states.shape[0], -1, -1),
                    ],
                    dim=1,
                )

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=None,
                causal_attention_mask=None,
            )

            hidden_states = layer_outputs[0]

        last_hidden_states = hidden_states
        pooled_output = last_hidden_states[:, 0, :]
        pooled_output = model.post_layernorm(pooled_output)

        visual_features = self.clip_model.visual_projection(pooled_output)

        return visual_features


    '''
    - Thực hiện toàn bộ quá trình tính toán, từ trích xuất đặc trưng hình ảnh, tạo đặc 
        trưng văn bản (sử dụng adaptive prompts từ LLM), tính toán logits, và 
        loss (nếu đang huấn luyện).

    '''
    def run(self, x, subset=None):
        # Chuẩn bị dữ liệu đầu vào.
        if self.training:
            img, img_1, labels = x
        else:
            img = x[0]

        # Điều chỉnh logit_scale.
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.605)

        normalize_fn = lambda x: F.normalize(x, dim=-1)
        logit_scale = self.logit_scale.exp()

        '''
        - Tạo/Lấy Class Embeddings:
            + Nếu đang huấn luyện, gọi self.class_fn (tức decode_class) để tạo adaptive
                prompts (embeds["llm"]) và lấy CLIP embeddings gốc (embeds["clip"]).
            + Nếu đang đánh giá, sử dụng self.all_class_embed đã được tính toán trước.
        '''
        embeds = {}

        if self.training:
            if self.few_shot:
                embeds["llm"], embeds["clip"] = self.class_fn(subset="all")[0]
            else:
                embeds["llm"], embeds["clip"] = self.class_fn(subset="base")[0]
        else:
            embeds["llm"], embeds["clip"] = self.all_class_embed

        embeds["all"] = embeds["llm"]

        raw_clip_embeds = embeds["clip"]
        raw_llm_embeds = embeds["llm"]

        # Chuẩn hóa tất cả các embeddings.
        for k, v in embeds.items():
            if embeds[k].ndim == 3:
                embeds[k] = normalize_fn(v).permute(0, 2, 1)
            else:
                embeds[k] = normalize_fn(v).permute(1, 0)

        if self.training:
            with torch.inference_mode():
                orig_image_features = self.clip_model.vision_model(img_1)[1]
                orig_image_features = self.clip_model.visual_projection(
                    orig_image_features
                )

        '''
            - Trích xuất Đặc trưng Hình ảnh: Gọi self.extract_image_features.
            - Tính toán Logits (Predictions):
                + Tính tích vô hướng (dot product) giữa đặc trưng hình ảnh đã chuẩn hóa và các
                    class embeddings (cả từ LLM và CLIP gốc) đã chuẩn hóa.
                + target_pred["all"]: Logits sử dụng adaptive prompts từ LLM.
                + target_pred["clip"]: Logits sử dụng CLIP embeddings gốc (dùng cho distillation loss).

        '''
        target_pred = {}
        if self.training:
            class_features = self.extract_image_features(img)
            image_features = class_features
            target_pred["clip"] = normalize_fn(image_features) @ embeds["clip"]
            if image_features.ndim != embeds["llm"].ndim:
                image_features = image_features.unsqueeze(0)
            target_pred["llm"] = torch.matmul(
                normalize_fn(image_features), embeds["llm"]
            )
            target_pred["all"] = torch.matmul(
                normalize_fn(image_features), embeds["all"]
            )
            clip_pred = normalize_fn(orig_image_features) @ embeds["clip"]

            target_pred["clip"] = target_pred["clip"].float()
            target_pred["llm"] = target_pred["llm"].float()
            target_pred["all"] = target_pred["all"].float()
            clip_pred = clip_pred.float()
            raw_clip_embeds = raw_clip_embeds.float()
            raw_llm_embeds = raw_llm_embeds.float()
            image_features = image_features.float()
            orig_image_features = orig_image_features.float()
        else:
            class_features = self.extract_image_features(img)
            image_features = class_features
            if image_features.ndim != embeds["llm"].ndim:
                image_features = image_features.unsqueeze(0)
            target_pred["all"] = torch.matmul(
                normalize_fn(image_features), embeds["all"]
            )
            target_pred["all"] = target_pred["all"].float()

        # Huấn luyện
        if self.training:
            # Tính toán Loss
            #   + base_loss: Cross-entropy loss giữa target_pred["all"] và nhãn.
            #   + feature_l1_loss: L1 loss giữa adaptive prompts từ LLM và CLIP embeddings 
            #       gốc, và L1 loss giữa đặc trưng hình ảnh từ LLaMP và đặc trưng hình ảnh 
            #       từ CLIP gốc.
            #   + dist_loss: Knowledge distillation loss (KL divergence hoặc cross-entropy) 
            #       giữa dự đoán từ LLaMP và dự đoán từ CLIP gốc.
            #   + loss_total = base_loss + feature_l1_loss + dist_loss.

            base_loss = self.base_loss(target_pred["all"] * logit_scale, labels)
            feature_l1_loss = (
                F.l1_loss(normalize_fn(raw_clip_embeds), normalize_fn(raw_llm_embeds))
                * 25
            )
            feature_l1_loss += (
                F.l1_loss(
                    normalize_fn(image_features), normalize_fn(orig_image_features)
                )
                * 10
            )

            if self.distillation_type == "soft":
                dist_loss = (
                    F.kl_div(
                        F.log_softmax(target_pred["all"] * logit_scale, dim=-1),
                        F.log_softmax(clip_pred * logit_scale, dim=-1),
                        reduction="sum",
                        log_target=True,
                    )
                    / target_pred["all"].numel()
                    * self.lambda_dist
                )
            elif self.distillation_type == "hard":
                dist_loss = (
                    F.cross_entropy(
                        target_pred["all"] * logit_scale,
                        clip_pred.argmax(dim=-1),
                        reduction="mean",
                    )
                    * self.lambda_dist
                )

            loss = base_loss + feature_l1_loss + dist_loss
            losses = {
                "loss_ce": base_loss,
                "loss_dist": dist_loss,
                "loss_l1": feature_l1_loss,
                "loss_total": loss,
            }
            
            # Trả về loss và dự đoán
            return losses, target_pred["all"]
        else:
            # Trả về dự đoán
            if target_pred["all"].ndim == 2:
                return target_pred["all"]
            else:
                return F.softmax(target_pred["all"].float(), dim=-1).mean(dim=0)
