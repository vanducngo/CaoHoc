'''
Import các thư viện cần thiết
'''

import torch
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
import torch.backends.cudnn as cudnn  # Tối ưu hóa cho GPU NVIDIA.

cudnn.benchmark = True

import os
import json
import copy
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

import deepspeed  # Thư viện để huấn luyện và suy luận mô hình lớn hiệu quả.
from flags import parser, DATA_FOLDER
from data.meta_dataset import MetaDataset # Module để tải và xử lý các tập dữ liệu meta (bao gồm base và new).
from models.common import Classification # Module để đánh giá phân loại.
from models.llamp import LLaMP # Module chứa định nghĩa của mô hình LLaMP.

from transformers import LlamaForCausalLM, LlamaTokenizer # Từ thư viện Hugging Face Transformers để tải mô hình LLaMA và tokenizer.
from transformers.integrations import HfDeepSpeedConfig # Tích hợp DeepSpeed với Hugging Face.
from dotenv import load_dotenv

# Tải các biến môi trường
load_dotenv()
hf_token = os.getenv("HF_TOKEN") # Token xác thực để tải các mô hình từ Hugging Face Hub
path_reults = os.getenv("PATH_RESULTS") # Đường dẫn để lưu file kết quả CSV.

def main():
    # Thêm các tham số cấu hình của DeepSpeed vào parser đã có
    local_parser = deepspeed.add_config_arguments(parser)
    local_parser.add_argument("--target_dataset", type=str)
    args = local_parser.parse_args()
    logpath = args.logpath
    dataset = args.dataset

    # Tạo thư mục logpath nếu nó chưa tồn tại.
    os.makedirs(logpath, exist_ok=True)

    # Đọc file cấu hình JSON của DeepSpeed => deepspeed_config/zero2_a100_40g.json
    with open(args.deepspeed_config, "r") as fp:
        deepspeed_config = json.load(fp)
    dschf = HfDeepSpeedConfig(deepspeed_config)

    # Tải mô hình LLaMA lên cpu => meta-llama/Llama-2-7b-chat-hf
    llama_model = LlamaForCausalLM.from_pretrained(
        args.model_base, device_map="cpu", token=hf_token
    )
    # Tải tokenizer tương ứng với mô hình LLaMA.
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_base, device_map="cpu", token=hf_token
    )

    # Khởi tạo MetaDataset cho tập base
    base_testset = MetaDataset(
        phase="val", dataset=dataset, num_shots=args.coop_num_shots, seed=args.coop_seed
    )

    # Khởi tạo MetaDataset cho tập new
    new_testset = MetaDataset(
        phase="test", dataset=dataset, num_shots=args.coop_num_shots, seed=args.coop_seed
    )

    # lassnames: Lưu trữ tên các lớp của tập base và new.
    classnames = {"base": base_testset.classnames, "new": new_testset.classnames}

    base_loader = torch.utils.data.DataLoader(
        base_testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers
    )
    new_loader = torch.utils.data.DataLoader(
        new_testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers
    )

    # Khởi tạo đối tượng Classification
    evaluator_base = Classification(args, base_testset.idx2label)
    evaluator_new = Classification(args, new_testset.idx2label)
    device = torch.device(args.device)

    # Load file checkpoint của từng dataset
    if os.path.isdir(args.load):
        ckpt_dir = args.load
    else:
        ckpt_dir = os.path.dirname(args.load)

    # Lấy danh sách các file có đuôi .t7 (định dạng checkpoint) trong thư mục ckpt_dir
    ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".t7")])
    print(f"Found {len(ckpt_files)} checkpoints in {ckpt_dir}")

    results = []

    # Với mỗi checkpoint file =>
    for ckpt_file in ckpt_files:
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        print(f"Loading checkpoint from: {ckpt_path}")
        
        # Khởi tạo lại mô hình LLaMP cho mỗi checkpoint.
        model = LLaMP(base_testset, classnames, args, llama_model, tokenizer, few_shot=False)

        try:
            # Tải state_dict (trọng số đã lưu) từ file checkpoint. 
            # map_location="cpu" đảm bảo trọng số được tải lên CPU trước.
            state_dict = torch.load(ckpt_path, map_location="cpu")
            # Tải trọng số vào mô hình
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Failed to load model from checkpoint {ckpt_path}: {e}")
            continue

        # Khởi tạo mô hình với DeepSpeed. 
        # DeepSpeed sẽ xử lý việc phân phối mô hình lên các GPU, tối ưu hóa bộ nhớ và tính toán.
        # trả về "model_engine" là đối tượng mô hình đã được DeepSpeed "bọc" lại.
        model_engine, _, _, _ = deepspeed.initialize(config=deepspeed_config, model=model)
        model_engine.eval() # Chuyển mô hình sang chế độ đánh giá (tắt dropout, batch normalization)

        # Tắt việc tính toán gradient, giúp tiết kiệm bộ nhớ và tăng tốc độ suy luận.
        with torch.no_grad():
            # Gọi hàm test() để đánh giá trên tập base và new.
            base_acc = test(0, model_engine, base_loader, evaluator_base, args, logpath, device, subset="Base")["accuracy"]
            new_acc = test(0, model_engine, new_loader, evaluator_new, args, logpath, device, subset="New")["accuracy"]
            
            # Harmonic Mean
            hm = 2 * base_acc * new_acc / (base_acc + new_acc)

            print(f"{ckpt_file}: Base: {base_acc:.4f}, New: {new_acc:.4f}, HM: {hm:.4f}")

            # Lưu vào danh sách results.
            results.append({
                "dataset": dataset,
                "checkpoint": ckpt_file,
                "Base": round(base_acc, 4),
                "New": round(new_acc, 4),
                "HM": round(hm, 4),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    if results:
        base_avg = np.mean([r["Base"] for r in results])
        new_avg = np.mean([r["New"] for r in results])
        hm_avg = np.mean([r["HM"] for r in results])
        results.append({
            "dataset": dataset,
            "checkpoint": "AVERAGE",
            "Base": round(base_avg, 4),
            "New": round(new_avg, 4),
            "HM": round(hm_avg, 4),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(path_reults, f"results_{dataset.lower()}.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Saved all results to {results_csv_path}")

def test(epoch, model, testloader, evaluator, args, logpath, device, subset):
    evaluator.reset() # Reset trạng thái của evaluator trước mỗi lần đánh giá.
    model.eval() # # Đảm bảo mô hình ở chế độ eval
    
    # Trước khi đánh giá ảnh, mô hình cần tính toán trước các vector embedding cho tất cả các
    # tên lớp trong tập subset (base hoặc new).
    model.module.compute_all_class_embeddings(subset=subset.lower())

    # Lặp qua từng batch dữ liệu trong testloader.
    for _, data in tqdm(
        enumerate(testloader), total=len(testloader), desc=f"Testing on {subset}"
    ):
        data = [d.to(device) for d in data]
        data[0] = data[0].bfloat16() # Image features
        data[1] = data[1].bfloat16() # Text features

        with torch.inference_mode():
            # Thực hiện suy luận
            _, predictions = model(data, subset=subset.lower())

        # predictions là đầu ra của mô hình
        predictions = predictions.cpu()
        evaluator.process(predictions, data[-1].cpu())

    stats = evaluator.evaluate()
    stats["a_epoch"] = epoch

    summary = " | ".join([f"{k}: {round(v, 4)}" for k, v in stats.items()])
    print(f"Test Epoch {epoch} [{subset}]: {summary}")
    return stats

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
