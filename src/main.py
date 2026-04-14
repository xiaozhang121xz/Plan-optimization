# 加载SFT之后的模型
# python -u /home/pc/matrad/zhangxue/SmolAgents/finally/agent2-4.py 2>&1 | tee /home/pc/matrad/zhangxue/SmolAgents/result2/logs/run_$(date +%Y%m%d_%H%M%S).log
import os
# 修复：替换废弃的CUDA显存配置变量
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"  # 替代原PYTORCH_CUDA_ALLOC_CONF
import torch
torch.cuda.empty_cache()

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from peft import PeftModel
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
from smolagents import tool
from typing import Any
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset
from datetime import datetime


# ========================== 配置（统一输出到 result 目录）==========================

DATA_FILE = "data_train_multi-patients/zuizhong2.mat"

MATLAB_PATH = " "

initial_dose = [40.0, 50.0, 38.0, 38.0, 40.0, 50.0, 15.0, 20.0, 70.0, 51.0, 52.4, 50.4]


BASE_RESULT_DIR = "result"
RESULT_DIR = os.path.join(BASE_RESULT_DIR, "dvh_plots")           # DVH 图
CLINICAL_REPORT_DIR = os.path.join(BASE_RESULT_DIR, "linchuangbaogao")  # 临床报告
GRPO_OUTPUT_DIR = os.path.join(BASE_RESULT_DIR, "grpo_output")    # GRPO 训练输出
LOG_DIR = os.path.join(BASE_RESULT_DIR, "logs")                   # 日志文件
LLM_OUTPUT_DIR = os.path.join(BASE_RESULT_DIR, "llm_outputs")     

os.makedirs(BASE_RESULT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CLINICAL_REPORT_DIR, exist_ok=True)
os.makedirs(GRPO_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(LLM_OUTPUT_DIR, exist_ok=True) 

print(f"所有结果将保存到: {BASE_RESULT_DIR}")
print(f"  - DVH 图: {RESULT_DIR}")
print(f"  - 临床报告: {CLINICAL_REPORT_DIR}")
print(f"  - GRPO 输出: {GRPO_OUTPUT_DIR}")
print(f"  - 日志: {LOG_DIR}")
print(f"  - LLM输出: {LLM_OUTPUT_DIR}") 

print("正在启动 MATLAB 引擎...")
eng = matlab.engine.start_matlab()
eng.addpath(MATLAB_PATH)
print("MATLAB 引擎已就绪")


# ========================== 全局步数计数器 ==========================
class StepCounter:
    """全局步数计数器，确保DVH图像和报告步数同步"""
    def __init__(self):
        self.current_step = 0
    
    def increment(self):
        self.current_step += 1
        return self.current_step
    
    def get(self):
        return self.current_step
    
    def reset(self):
        self.current_step = 0

step_counter = StepCounter()


# ========================== 工具函数（仅修改动作范围）==========================
@tool
def init_matrad_plan() -> str:
    """
    初始化 matRad 放疗计划（必须第一步调用）。

    Args:
        无参数

    Returns:
        str: 初始化成功消息，包含12个器官名称。
             示例: "初始化成功！共12个器官：PTV, Rectum, Bladder, ..."
    """
    global eng, step_counter
    try:
        step_counter.reset() 
        res = eng.reset_fda_singlepatient_Z75(DATA_FILE, matlab.double(initial_dose), nargout=11)
        (dvh_all, target_area, oar_area, _, dij, pln, _, organ_names_raw,
         clin_goals_raw, area_all_oar, dose_) = res  

        state = {
            "dij": dij,
            "pln": pln,
            "prev_target_area": float(target_area),
            "prev_oar_area": float(oar_area),
            "prev_oar_vec": np.array(area_all_oar).flatten().tolist(),
            "prev_constraints": np.array(dose_).flatten().tolist(),  # 新增：保存初始约束
            "organ_names": [str(x) for x in organ_names_raw],
            "step": 0
        }
        init_matrad_plan.state = state
        
        # 保存初始化的DVH图（步骤0）
        save_dvh_plot(np.array(dvh_all), state["organ_names"], 0)
        
        return f"初始化成功！共12个器官：{', '.join(state['organ_names'])}"
    except Exception as e:
        return f"初始化失败：{e}"


@tool
def execute_dose_action(action: list) -> dict:
    """
        执行一次剂量优化动作。

        Args:
            action (list): 长度为12的浮点数列表，代表12个器官的约束调整因子。
                        建议每个值在 [0.7, 1.7] 之间。
                        - 对于靶区（如PTV）：>1.0 表示推高剂量，<1.0 表示放宽
                        - 对于OAR：         <1.0 表示约束更严格，>1.0 表示放宽约束

        Returns:
            dict: 包含 reward、step、dvh_path、message 等信息。
                示例: {"reward": 12.34, "step": 5, "message": "第 5 步完成，奖励 = 12.340"}
        """
    global eng, step_counter
    if not hasattr(init_matrad_plan, 'state'):
        return {"error": "请先调用 init_matrad_plan", "reward": -1e9}

    state = init_matrad_plan.state
    state["step"] += 1
    step = state["step"]
    step_counter.current_step = step

    action = [max(0.9, min(1.1, float(a))) for a in action]
    print(f"\n第 {step} 步 | 执行动作: {np.round(action, 4)}")

    try:
        result = eng.area_conv_multi_Z75(  
            DATA_FILE, state["dij"], state["pln"], matlab.double(action),
            matlab.double(state["prev_constraints"]),
            matlab.double([state["prev_target_area"]]),
            matlab.double([state["prev_oar_area"]]),
            matlab.double(state["prev_oar_vec"]),
            nargout=14
        )

        (dvh_all, reward, dose_, target_area, oar_area, _, _, oar_vec,
         _, _, _, _, names, goals) = result

        reward = float(reward)
        state["prev_target_area"] = float(target_area)
        state["prev_oar_area"] = float(oar_area)
        state["prev_oar_vec"] = np.array(oar_vec).flatten().tolist()
        state["prev_constraints"] = np.array(dose_).flatten().tolist()

        # 保存 DVH 图
        dvh_path = save_dvh_plot(np.array(dvh_all), state["organ_names"], step)
        
        return {
            "reward": reward,
            "step": step,
            "dvh_path": dvh_path,
            "message": f"第 {step} 步完成，奖励 = {reward:.3f}"
        }
    except Exception as e:
        return {"error": str(e), "reward": -1e9}



@tool
def save_dvh_plot(dvh_matrix: list, organ_names: list, step: int) -> str:
    """绘制并保存当前 DVH 曲线（仅绘制指定6个器官）
    
    Args:
        dvh_matrix (list): DVH 数据矩阵，形状为 (12, 2000)，前1000列为剂量，后1000列为体积百分比
        organ_names (list): 器官名称列表，包含12个器官名称
        step (int): 当前优化步数，用于文件名和标题
    
    Returns:
        str: 保存的图片路径
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 【核心修改1】定义需要绘制的目标器官列表
    target_organs = [
        'Bladder',
        'Femoral Head R',
        'Femoral Head L',
        'Rectum',
        'Small intestine',
        'PTV_plan'
    ]
    
    dvh_array = np.array(dvh_matrix)
    dose = dvh_array[0, :1000]  # 所有器官共用同一剂量网格
    
    plt.figure(figsize=(11, 7))
    # 【核心修改2】只为目标器官生成颜色（避免颜色浪费，曲线更清晰）
    colors = plt.cm.tab20(np.linspace(0, 1, len(target_organs)))
    color_idx = 0  # 手动控制颜色索引
    
    # 【核心修改3】只遍历目标器官，跳过其他器官
    for i, name in enumerate(organ_names):
        # 过滤：仅处理目标器官
        if name not in target_organs:
            continue
        
        vol = dvh_array[i, 1000:]
        # 体积值转换为百分比（兼容原始数据格式）
        if vol.max() <= 1.0 + 1e-6:
            vol = vol * 100
        
        # 绘制目标器官的DVH曲线
        plt.plot(dose, vol, label=name, color=colors[color_idx], linewidth=2.3)
        color_idx += 1  # 仅目标器官递增颜色
    
    # 标题/标签优化：明确显示仅绘制指定器官
    plt.title(f"DVH Curve (Target Organs) - Step {step:03d}", fontsize=16)
    plt.xlabel("Dose (Gy)", fontsize=12)
    plt.ylabel("Volume (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    # 图例仅显示目标器官，位置优化
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    plt.tight_layout()

    # 保存路径（文件名可加标识，区分全器官图）
    path = os.path.join(RESULT_DIR, f"DVH_target_organs_step_{step:03d}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[DVH] Target organs DVH plot saved → {path}")
    return path


@tool
def get_latest_clinical_report() -> str:
    """获取最新临床报告文本"""
    files = [f for f in os.listdir(CLINICAL_REPORT_DIR) if f.startswith('report_step_') and f.endswith('.txt')]
    if not files:
        return "无报告可用"
    
    files.sort(key=lambda x: int(re.search(r'report_step_(\d+)', x).group(1)))
    latest_file = os.path.join(CLINICAL_REPORT_DIR, files[-1])
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        report_text = f.read()
    
    # 追加 DVH 图路径
    step_num = int(re.search(r'report_step_(\d+)', latest_file).group(1))
    dvh_path = os.path.join(RESULT_DIR, f'dvh_step_{step_num:03d}.png')
    report_text += f"\n\nDVH 图路径: {dvh_path}"
    
    return report_text


# ========================== LLM输出日志器 ==========================
class LLMOutputLogger:
    """记录LLM输出到会话目录"""
    def __init__(self, base_dir: str):
        self.session_dir = os.path.join(base_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.session_dir, exist_ok=True)
        self.summary_file = os.path.join(self.session_dir, "summary.md")
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write("# LLM 输出汇总\n\n")
    
    def log_step(self, step: int, user_input: str, llm_response: str, action: list, reward: float, dvh_path: str):
        step_file = os.path.join(self.session_dir, f"step_{step:03d}.md")
        with open(step_file, 'w', encoding='utf-8') as f:
            f.write(f"## 步骤 {step}\n\n")
            f.write("### 用户输入（临床报告）\n```\n" + user_input + "\n```\n\n")
            f.write("### LLM 输出\n```\n" + llm_response + "\n```\n\n")
            f.write(f"### 提取动作\n{action}\n\n")
            f.write(f"### 奖励\n{reward:.4f}\n\n")
            f.write(f"### DVH 图\n![DVH]({os.path.relpath(dvh_path, self.session_dir)})\n")
        
        with open(self.summary_file, 'a', encoding='utf-8') as f:
            f.write(f"## 步骤 {step}\n\n")
            f.write(f"奖励: {reward:.4f}\n\n")
            f.write(f"[查看详情](step_{step:03d}.md)\n\n")
    
    def save_final_summary(self, rewards: list):
        with open(self.summary_file, 'a', encoding='utf-8') as f:
            f.write("## 最终统计\n\n")
            f.write(f"总步数: {len(rewards)}\n")
            f.write(f"最终奖励: {rewards[-1]:.4f}\n")
            f.write(f"最佳奖励: {max(rewards):.4f}\n")
            f.write(f"平均奖励: {np.mean(rewards):.4f}\n")
            f.write(f"奖励标准差: {np.std(rewards):.4f}\n")


# ========================== 动作提取函数 ==========================
def extract_action(text: str) -> list:
    # 优先匹配 <最终动作> 标签
    match = re.search(r'<最终动作>\s*\[([\d.,\s]+)\]\s*</最终动作>', text)
    if match:
        # 修改点：支持空格和逗号分隔
        nums = [float(x.strip()) for x in re.split(r'[\s,]+', match.group(1).strip()) if x.strip()]
    else:
        # 备选：匹配任意 [] 内数字序列
        match = re.search(r'\[([\d.,\s]+)\]', text)
        if match:
            # 修改点：同样支持空格和逗号分隔
            nums = [float(x.strip()) for x in re.split(r'[\s,]+', match.group(1).strip()) if x.strip()]
        else:
            nums = []
    
    # 只取前9个，后3个补1.0
    nums = nums[:9] + [1.0] * (12 - len(nums[:9]))
    if len(nums) < 12:
        nums += [1.0] * (12 - len(nums))
    
    # 限制范围 [0.9, 1.1]
    nums = [max(0.9, min(1.1, x)) for x in nums]
    return nums


# ========================== 自定义 GRPOTrainer （添加钩子捕获内部生成） ==========================
class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generated_completions = []
        self.generated_rewards = []

    def train(self):
        print("[DEBUG] Custom train 被调用，开始清空列表")
        self.generated_completions = []
        self.generated_rewards = []
        super().train()
        print(f"[DEBUG] train 结束，completions 数量: {len(self.generated_completions)}")
        return self.generated_completions, self.generated_rewards

    # 核心 hook：重写 _generate_completions（源码中生成 rollout 的直接入口）
    def _generate_completions(self, batch: dict, **kwargs):
        print("[GRPO HOOK] _generate_completions 被调用！强制长输出 + SFT adapter")

        # 强制切换到 SFT adapter（生成用成熟行为）
        self.model.set_adapter("default")  # 如果 SFT adapter 不是 "default"，改成实际名字

        # 强制覆盖所有生成参数
        forced_generation_kwargs = {
            "max_new_tokens": 1024,
            "min_new_tokens": 400,            # 强制至少 400 token
            "temperature": 0.4,
            "top_p": 0.92,
            "repetition_penalty": 1.05,
            "do_sample": True,
            "eos_token_id": None,             # 禁用早停（最关键！防止截断）
            "pad_token_id": self.tokenizer.pad_token_id,
            "output_logits": True,            # 保持源码需要
        }

        print("[GRPO HOOK] 强制 generation_kwargs:", forced_generation_kwargs)

        # 调用父类的 _generate_completions，但传入我们的参数
        rollout_outputs = super()._generate_completions(
            batch,
            **kwargs,
            generation_kwargs=forced_generation_kwargs,  # 覆盖默认 kwargs
        )

        # 捕获生成的 completions（源码返回的 dict 中有）
        completions = rollout_outputs["completions"]
        self.generated_completions.extend(completions)

        # 生成完切换回 GRPO adapter（用于后续 loss 计算和更新）
        self.model.set_adapter("grpo_adapter")

        return rollout_outputs


# ========================== GRPO 核心代理类（完整修复）==========================
class GRPOAgent:
    def custom_rollout_func(prompts: list[str], trainer: "GRPOTrainer") -> dict[str, Any]:
        print("[CUSTOM ROLLOUT] custom_rollout_func 被调用！生成 rollout 样本")

        # 强制切换到 SFT adapter（使用训练好的生成行为）
        trainer.model.set_adapter("default")

        # 准备 tokenized prompts
        tokenized_prompts = trainer.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=trainer.args.max_prompt_length or 512,
        ).to(trainer.model.device)

        # 自定义生成参数（关键：这里可以控制长度、禁用早停等）
        generation_kwargs = {
            "max_new_tokens": 1024,
            "min_new_tokens": 400,            # 强制至少生成 400 token
            "temperature": 0.4,
            "top_p": 0.92,
            "repetition_penalty": 1.05,
            "do_sample": True,
            "eos_token_id": None,             # 禁用 EOS 早停
            "pad_token_id": trainer.tokenizer.pad_token_id,
        }

        print("[CUSTOM ROLLOUT] 使用参数:", generation_kwargs)

        # 调用模型生成
        outputs = trainer.model._generate(**tokenized_prompts, **generation_kwargs)

        # 解码生成的 completions（去掉 prompt 部分）
        completions = trainer.tokenizer.batch_decode(
            outputs[:, tokenized_prompts["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # 返回 trl 期望的格式
        return {
            "prompts": prompts,
            "completions": completions,
            # 如果需要 logits，可加 output_logits=True 并返回 outputs.logits
        }

    def __init__(self, model_path="/home/pc/matrad/zhangxue/models/Qwen3-0.6B"):
        # 新增：存储上一步的数据
        self.previous_system_prompt = None
        self.previous_report = None
        self.previous_completion = None
        
        sft_lora_path = "/home/pc/matrad/zhangxue/models/SFT/Qwen3-0.6B"
        ##1.加载 Tokenizer 和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id
        self.tokenizer.eos_token_id = self.tokenizer.eos_token_id
            ##使用 4bit 量化，降低显存占用
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        print("正在加载基础模型（4bit）...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )

        print(f"正在加载 SFT 后的 LoRA 适配器：{sft_lora_path}")
        # 先把 SFT 训练好的 LoRA 挂载上去（不合并）
        model = PeftModel.from_pretrained(
            base_model,
            sft_lora_path,
            device_map="auto",
            dtype=torch.bfloat16,
        )

        # 再在此基础上新增一个用于 GRPO 的 LoRA 适配器
        print("正在创建用于 GRPO 的新 LoRA 层...")
        grpo_peft_config = LoraConfig(
            r=64,                     # 可以保持和 SFT 一样，也可以调小（如32、16）来进一步省显存
            lora_alpha=16,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            # 可选：给这个适配器起个名字，方便后续管理（非必须）
            # modules_to_save=["lm_head", "embed_tokens"]   # 如果你 SFT 时有保存这些，也可以在这里加
        )

        # 叠加新的 LoRA
        model.add_adapter(
        peft_config=grpo_peft_config,
        adapter_name="grpo_adapter"
        )

        # 激活刚刚添加的这个 adapter（非常重要！）
        #model.set_adapter("grpo_adapter")

        self.model = model

        # 建议打印一下，看看是否只有新加的 LoRA 在训练
        self.model.print_trainable_parameters()

        self.model.train()  # 切换到训练模式（仅影响Dropout等层，不修改梯度）
        self.model.print_trainable_parameters()  # 验证仅Lora层可训练

        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.bos_token_id = self.tokenizer.bos_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id

         # 3 配置 GRPO 训练参数
        self.grpo_config = GRPOConfig(
            per_device_train_batch_size=1,   ##每次真正算梯度用几条数据
            gradient_accumulation_steps=1,
            learning_rate=5e-7,        #极小学习率（5e-7），适合在线微调，避免灾难性遗忘。
            num_generations=2,        ## 每次训练生成 4 个候选回复，用外部奖励评估         
            generation_batch_size=4,  #### 
            num_iterations=1,
            epsilon=0.2,
            beta=0.01,
            max_grad_norm=1.0,
            output_dir=GRPO_OUTPUT_DIR,  # 【修改】统一到 result 目录
            logging_dir=LOG_DIR,          # 【新增】日志目录
            logging_steps=1,
            torch_compile=False,
            optim="adamw_torch",
            fp16=False,#True             # 关闭 fp16 AMP
            bf16=True,
            #max_steps=1,
            num_train_epochs=1,
        )
        # 建议：明确记录我们打算采样几条，便于后续阅读和调试
        self.num_sample_per_step = 1 

        # ============ 外部奖励函数============
        #4.定义外部奖励函数 external_reward，奖励不是来自 KL 散度或人工打分，而是来自真实的物理剂量优化模拟。
        def external_reward(prompts, completions, completion_ids=None, **kwargs):
            """
            适配新版trl的奖励函数签名
            
            Args:
                prompts: 提示列表
                completions: 生成的回复列表
                completion_ids: 生成的token id列表（可选）
                **kwargs: 其他参数（如 trainer_state 等）
            Returns:
                list: 与completions等长的奖励列表
            """
            global eng
            print("[DEBUG] external_reward 被调用，捕获到以下 completions：")

            for i, comp in enumerate(completions, 1):
                print(f"  候选 {i} (长度 {len(comp)} chars):")
                print(comp)                     # 直接完整打印
                print("-" * 80)                 # 用更长的分隔线区分更清晰

            reward_list = []
            # 获取当前放疗计划状态
            state = init_matrad_plan.state if hasattr(init_matrad_plan, 'state') else None
            
            if not state or not completions:
                # 未初始化/无候选回复时返回兜底奖励
                return [0] * len(completions)
            
            # 遍历每个候选回复，独立计算奖励
            for idx, completion in enumerate(completions):
                try:
                    # 提取当前候选回复的动作
                    action = extract_action(completion)
                    # 调用MATLAB计算该动作的奖励
                    result = eng.area_conv_multi_Z75(
                        DATA_FILE, state["dij"], state["pln"], matlab.double(action),
                        matlab.double(state["prev_constraints"]),
                        matlab.double([state["prev_target_area"]]),
                        matlab.double([state["prev_oar_area"]]),
                        matlab.double(state["prev_oar_vec"]),
                        nargout=14
                    )
                    # 提取奖励值
                    (_, reward, _, _, _, _, _, _, _, _, _, _, _, _) = result
                    # 避免负奖励，加基础值10
                    reward = float(reward)
                    reward_list.append(reward)
                    print(f"【候选回复{idx+1}】提取动作：{np.round(action,4)} | 计算奖励：{reward:.4f}")
                except Exception as e:
                    print(f"【候选回复{idx+1}】奖励计算失败：{e}，使用兜底奖励0.0")
                    reward_list.append(0)
            
            # 确保奖励列表长度匹配
            if len(reward_list) != len(completions):
                reward_list = [0] * len(completions)##？改成0
            return reward_list
    
        # 创建一个空的占位数据集（GRPOTrainer 需要 train_dataset）
        # 我们会在 step() 中动态替换它
        placeholder_dataset = Dataset.from_list([{"prompt": "placeholder"}])

        #  #5.创建GRPOTrainer（Trl）
        self.grpo_trainer = CustomGRPOTrainer(  # 使用自定义 Trainer
            model=self.model,
            args=self.grpo_config,
            reward_funcs=[external_reward],# #############GRPO 将调用这个函数获取真实物理奖励来更新策略############
            train_dataset=placeholder_dataset,
            processing_class=self.tokenizer,
            rollout_func=self.custom_rollout_func,
        )

        self.rewards = []
        
        # 【新增】初始化LLM输出日志器
        self.llm_logger = LLMOutputLogger(LLM_OUTPUT_DIR)
        
        print("GRPOAgent 初始化完成（真实 GRPO + 外部物理 reward）")
                # 确保默认使用 SFT adapter
        self.model.set_adapter("default")
        print("[INFO] 默认使用 SFT adapter 进行生成")
    def generate(self, messages):
        print("[DEBUG] Custom generate 被调用了！")
        
        # 1. 应用聊天模板
        try:
            query = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,   # 建议开启，确保生成时有 assistant prefix
                enable_thinking=True
            )
            print("[generate] apply_chat_template 完成，query 长度（字符）:", len(query))
        except Exception as e:
            print("[ERROR] apply_chat_template 失败:", str(e))
            raise

        # 2. tokenization
        try:
            inputs = self.tokenizer(query, return_tensors="pt").to("cuda")
            print("[generate] inputs 准备完成，input_ids shape:", inputs.input_ids.shape)
        except Exception as e:
            print("[ERROR] tokenizer 失败:", str(e))
            raise

        # 3. 清缓存 + 打印显存
        torch.cuda.empty_cache()
        print("[generate] 已清空缓存，当前显存使用:", torch.cuda.memory_summary(abbreviated=True))
        print("[generate] 已分配显存 (GB):", torch.cuda.memory_allocated() / 1024**3)

        # 4. 切换到 eval 模式
        self.model.eval()

        with torch.no_grad():
            print("[generate] 即将调用 model.generate...")
            
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,           # 降低到 1024，先测试链路通不通（可逐步提高）
                    min_new_tokens=200,            # 强制至少生成 200 个新 token，避免早停
                    do_sample=True,
                    temperature=0.4,               # 稍提高一点，避免过于确定性导致早停
                    top_p=0.92,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,  # 保留，让模型能正常停；如需禁用改成 None
                    
                    # 如果想强制禁用 EOS（不推荐，除非测试）：
                    # eos_token_id=None,
                    
                    # 可选：其他常用参数
                    # num_beams=1,                 # 如果 greedy 更快，可加 beam search
                    # early_stopping=True,
                )
                
                print("[generate] model.generate 成功完成！")
                print("[generate] outputs shape:", outputs.shape)
                print("[generate] 生成的新 token 数量:", outputs.shape[1] - inputs.input_ids.shape[1])
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("[OOM ERROR] 显存不足！建议：降低 max_new_tokens 或关闭 4bit 量化")
                print("[RuntimeError] model.generate 失败:", str(e))
                import traceback
                traceback.print_exc()
                raise
            except Exception as e:
                print("[ERROR] model.generate 其他异常:", str(e))
                import traceback
                traceback.print_exc()
                raise

        # 5. 解码输出
        try:
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            print("[generate] 解码完成，response 长度（字符）:", len(response))
            return response
        except Exception as e:
            print("[ERROR] decode 失败:", str(e))
            raise

    def step(self, report_text: str, step_num: int):
        """
        修改后核心优化步骤：
        - 如果是更新步（%5==0）：手动生成 2 个候选回复，计算奖励，选择奖励最高的作为主动作执行（保存DVH/报告），然后用 2 个候选更新 GRPO。
        - 非更新步：原样生成 1 个回复，直接执行。
        """
        system_prompt = """
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"【第{step_num}步临床报告】\n{report_text}\n\n请分析并输出12个器官的优化动作。"}
        ]

        # 构建当前 prompt 字符串
        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

        # 构建数据集的 prompt（包含上一步数据）
        if self.previous_system_prompt and self.previous_report and self.previous_completion:
            dataset_prompt = f"上一步 system prompt: {self.previous_system_prompt}\n上一步临床报告: {self.previous_report}\n上一步 completion: {self.previous_completion}\n当前 prompt: {prompt_str}"
        else:
            dataset_prompt = prompt_str

        # 创建只包含 prompt 的 dataset
        #train_dataset = Dataset.from_list([{"prompt": dataset_prompt}])
        from datasets import Dataset

        train_dataset = Dataset.from_dict({
            "prompt": [dataset_prompt] * 2   # 直接列表乘法
        })

        self.grpo_trainer.train_dataset = train_dataset

        print(f"\n{'='*70}")
        print(f"[Step {step_num}] 开始 GRPO 训练（生成 4 个候选回复）...")
        
        self.model.train()
        # 执行训练，并捕获内部生成
        self.model.set_adapter("default")  #
        generated_completions, generated_rewards = self.grpo_trainer.train()

        # 选 reward 最高的执行
        if generated_rewards:
            best_idx = np.argmax(generated_rewards)
            best_reward = generated_rewards[best_idx]
            best_completion = generated_completions[best_idx]
            best_action = extract_action(best_completion)
            
            print(f"[Step {step_num}] 选择最佳动作（第 {best_idx+1} 条，reward = {best_reward:.4f}）")
        else:
            best_action = [1.0] * 12
            best_reward = -1e9
            best_completion = "生成失败，使用默认动作"
            print("[警告] 生成失败，使用默认动作")

        print(f"[Step {step_num}] 执行动作: {np.round(best_action, 4)}")

        # 真正执行
        exec_result = execute_dose_action(best_action)
        
        if "error" in exec_result:
            print(f"[执行失败] {exec_result['error']}")
            final_reward = -1e9
            dvh_path = None
        else:
            final_reward = exec_result["reward"]
            dvh_path = exec_result.get("dvh_path")
            print(f"[执行成功] 最终奖励 = {final_reward:.4f}")

        self.rewards.append(final_reward)

        # 更新上一.step 数据
        self.previous_system_prompt = system_prompt
        self.previous_report = report_text
        self.previous_completion = best_completion

        # 日志记录
        self.llm_logger.log_step(
            step=step_num,
            user_input=report_text,
            llm_response=best_completion,
            action=best_action,
            reward=final_reward,
            dvh_path=dvh_path
        )

        return {
            "step": step_num,
            "action": best_action,
            "reward": final_reward,
            "llm_response": best_completion,
            "dvh_path": dvh_path
        }

    def _grpo_update(self, messages, candidates, reward):
        """修改后：用手动生成的 candidates（2 个）构建数据集更新 GRPO"""
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                #add_generation_prompt=True,
                enable_thinking=True
            )
            
            # 构建包含 2 个样本的数据集（prompt + completion）
            train_list = [{"prompt": prompt, "completion": cand["llm_response"]} for cand in candidates]
            train_data = Dataset.from_list(train_list)
            self.grpo_trainer.train_dataset = train_data
            print(f"[GRPO update] 本次训练数据集大小: {len(train_data)}  (candidates 数量: {len(candidates)})")
            # 执行训练（TRL 会自动调用 external_reward 计算奖励）
            self.grpo_trainer.train()
            print(f"[GRPO] 策略已更新，当前奖励 = {reward:.4f}")
        except Exception as e:
            print(f"[GRPO] 更新失败（不影响主流程）: {e}")



    def run_optimization(self, max_steps: int = 250):
        """
        运行完整的优化流程
        
        Args:
            max_steps: 最大优化步数
        """
        print("\n" + "="*80)
        print("开始放疗计划优化流程")
        print("="*80 + "\n")

        # 1. 初始化放疗计划
        init_result = init_matrad_plan()
        print(f"初始化结果: {init_result}")

        # 2. 循环优化
        for step in range(1, max_steps + 1):
            print(f"\n{'#'*80}")
            print(f"### 优化步骤 {step}/{max_steps} ###")
            print(f"{'#'*80}")

            # 获取最新临床报告作为LLM输入
            report = get_latest_clinical_report()
            
            # 执行一步优化
            result = self.step(report, step)
            
            # 打印进度
            print(f"\n[进度] 步骤 {step} 完成")
            print(f"  - 奖励: {result['reward']:.4f}")
            print(f"  - DVH图: {result['dvh_path']}")
            
            # 早停条件（可选）
            #if result['reward'] > 100:  # 根据实际情况调整阈值
                #print(f"\n[早停] 奖励达到目标值 {result['reward']:.4f}，停止优化")
                #break

        # 3. 保存最终汇总
        self.llm_logger.save_final_summary(self.rewards)
        
        # 4. 绘制奖励曲线
        self._plot_reward_curve()
        
        print("\n" + "="*80)
        print("优化流程完成！")
        print(f"  - 总步数: {len(self.rewards)}")
        print(f"  - 最终奖励: {self.rewards[-1]:.4f}")
        print(f"  - 最佳奖励: {max(self.rewards):.4f}")
        print(f"  - 所有结果保存在: {BASE_RESULT_DIR}")
        print("="*80 + "\n")

        return self.rewards

    def _plot_reward_curve(self):
        """绘制并保存奖励曲线"""
        plt.figure(figsize=(12, 6))
        
        # 主曲线
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(self.rewards)+1), self.rewards, 'b-o', linewidth=2, markersize=4)
        plt.axhline(y=max(self.rewards), color='r', linestyle='--', label=f'Best: {max(self.rewards):.2f}')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title('Optimization Reward Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 滑动平均
        plt.subplot(1, 2, 2)
        window = min(5, len(self.rewards))
        if window > 1:
            moving_avg = np.convolve(self.rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window, len(self.rewards)+1), moving_avg, 'g-', linewidth=2, label=f'{window}-step Moving Avg')
        plt.plot(range(1, len(self.rewards)+1), self.rewards, 'b-', alpha=0.3, label='Raw')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title('Reward with Moving Average')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        curve_path = os.path.join(BASE_RESULT_DIR, "reward_curve.png")
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[奖励曲线] 已保存 → {curve_path}")



# ========================== 主程序入口 ==========================
def main():
    """主函数"""
    print("\n" + "="*80)
    print("放疗计划智能优化系统 v2.0")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出目录: {BASE_RESULT_DIR}")
    print("="*80 + "\n")

    try:
        # 初始化Agent
        agent = GRPOAgent()

        # 运行优化（可调整步数）
        rewards = agent.run_optimization(max_steps=250)
        
        # 打印最终统计
        print("\n【最终统计】")
        print(f"  总步数: {len(rewards)}")
        print(f"  最终奖励: {rewards[-1]:.4f}")
        print(f"  最佳奖励: {max(rewards):.4f}")
        print(f"  平均奖励: {np.mean(rewards):.4f}")
        print(f"  奖励标准差: {np.std(rewards):.4f}")
        
    except KeyboardInterrupt:
        print("\n[中断] 用户手动停止")
    except Exception as e:
        print(f"\n[错误] 程序异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭MATLAB引擎
        try:
            eng.quit()
            print("[清理] MATLAB引擎已关闭")
        except:
            pass


if __name__ == "__main__":
    main()